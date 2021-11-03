import argparse
import copy
import os
import pdb
import random
import sys
from collections import OrderedDict

import numpy as np
import torch

from anil import ANIL
from data_generator import MiniImagenet, MiniImagenet_MW
from scheduler import Scheduler

parser = argparse.ArgumentParser(description='graph transfer')
parser.add_argument('--datasource', default='miniImagenet', type=str,
                    help='sinusoid or omniglot or miniimagenet or mixture or metadataset or metadataset_leave_one_out or multiscale')
parser.add_argument('--select_data', default=-1, type=int, help='-1,0,1,2,3')
parser.add_argument('--test_dataset', default=-1, type=int,
                    help='which dataset to be test: 0: bird, 1: texture, 2: aircraft, 3: fungi, -1 is test all')
parser.add_argument('--num_classes', default=5, type=int,
                    help='number of classes used in classification (e.g. 5-way classification).')
parser.add_argument('--num_test_task', default=600, type=int, help='number of test tasks.')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')

## Training options
parser.add_argument('--metatrain_iterations', default=20000, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', default=2, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in anil')
parser.add_argument('--num_updates_test', default=10, type=int, help='num_updates in anil')
parser.add_argument('--update_batch_size', default=1, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')
parser.add_argument('--num_filters', default=32, type=int,
                    help='number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

## Logging, saving, and testing options
parser.add_argument('--log', default=1, type=int, help='if false, do not log summaries, for debugging code.')
parser.add_argument('--logdir', default='../train_logs', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--datadir', default='../../miniimagenet', type=str, help='directory for datasets.')

parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
parser.add_argument('--test_set', default=1, type=int,
                    help='Set to true to test on the the test set, False for the validation set.')
parser.add_argument('--trail', default=0, type=int, help='trail for each layer')
parser.add_argument('--buffer_size', type=int, default=6)
parser.add_argument('--sampling_method', default=None)
parser.add_argument('--trace', default=0, help='whether to trace the learning curve')
parser.add_argument('--noise', default=0.0, type=float, help='noise ratio')
parser.add_argument('--replace', default=0, type=int, help='whether to allow sampling the same task, 1 for allow')
parser.add_argument('--limit_data', default=False, type=int, help='whether to use limited data to do the experiments')
parser.add_argument('--out_learner', default='anil', type=str)
parser.add_argument('--tb_dir', default='./tensorboard', type=str)
parser.add_argument("--device", default='cuda')
parser.add_argument('--mix', default=False, type=int)
parser.add_argument("--grad_type", default='norm_cos', type=str)
parser.add_argument("--lambda0", default=1.0, type=float)
parser.add_argument("--limit_classes", default=16, type=int)
parser.add_argument("--debug", action="store_true", default=False)
parser.add_argument("--finetune", action='store_true', default=False)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--scheduler_lr", type=float, default=0.001)
parser.add_argument("--t_decay", type=int, default=True)
parser.add_argument("--warmup", type=int, default=2000)
parser.add_argument("--pretrain_iter", type=int, default=20000)
parser.add_argument("--seed", default=1)

args = parser.parse_args()
print(args)

assert torch.cuda.is_available()
torch.backends.cudnn.benchmark = True

def set_seed(seed):
    """Sets seed"""
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)

exp_string = f'{args.out_learner.upper()}_pytorch' + '.data_' + str(args.datasource) + 'cls_' + str(args.num_classes) + '.mbs_' + str(
    args.meta_batch_size) + '.ubs_' + str(
    args.update_batch_size) + '.metalr' + str(args.meta_lr) + '.innerlr' + str(args.update_lr)

if args.num_filters != 64:
    exp_string += '.hidden' + str(args.num_filters)
if args.trail > 0:
    exp_string += '.trail{}'.format(args.trail)
if args.sampling_method is not None:
    exp_string += '.Sample-{}'.format(args.sampling_method)
    if args.sampling_method == 'ATS':
        exp_string += f'_pool{args.buffer_size}'
        if args.temperature != 1:
            exp_string += f'_temperature_{args.temperature}'
        exp_string += f'_schedulerlr_{args.scheduler_lr}'
if args.noise > 0:
    exp_string += f'_noise{args.noise}'
if args.limit_data:
    exp_string += f'_limit_data_{args.limit_classes}'
if args.finetune:
    exp_string += '_finetune'


print(exp_string)


def weight_norm(task_weight):
    sum_weight = torch.sum(task_weight)
    assert sum_weight > 0
    if sum_weight == 0:
        sum_weight = 1.0
    task_weight = task_weight / sum_weight

    return task_weight



def get_inner_loop_parameter_dict(params):
    """
    Returns a dictionary with the parameters to use for inner loop updates.
    :param params: A dictionary of the network's parameters.
    :return: A dictionary of the parameters to use for the inner loop optimization process.
    """
    param_dict = dict()
    indexes = []
    for i, (name, param) in enumerate(params):
        if param.requires_grad:
            if "1.weight" in name or '1.bias' in name: continue
            param_dict[name] = param.to(device=args.device)
            indexes.append(i)

    return param_dict, indexes


def update_moving_avg(mv, reward, count):
    return mv + (reward.item() - mv) / (count + 1)


def train(args, anil, optimiser):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    train_steps = [1]
    train_losses = []
    train_probs = []
    train_clean_probs = []
    train_noisy_probs = []
    train_clean_losses = []
    train_noisy_losses = []
    train_accs = []

    if args.datasource.lower() == 'miniimagenet':
        if args.finetune:
            if args.sampling_method == 'ATS':
                name = 'ATS'
            elif args.sampling_method is None:
                name = 'ANIL'
            else:
                name = args.sampling_method
            found = False
            for file in os.listdir("./models/"):
                if file.startswith(f"{name}_{args.noise}_model_{args.update_batch_size}shot"):
                    found = True
                    break
            assert found
            anil.load_state_dict(torch.load(f"./models/{file}"))
            print("Load model successfully")
            dataloader = MiniImagenet(args, 'val')

            args.sampling_method = None
            args.DAML = False
            args.FocalLoss = False

        else:
            if args.sampling_method == 'ATS':
                dataloader = MiniImagenet_MW(args, 'train')
            else:
                dataloader = MiniImagenet(args, 'train')

    if not os.path.exists(args.logdir + '/' + exp_string + '/'):
        os.makedirs(args.logdir + '/' + exp_string + '/')

    if args.sampling_method == 'ATS':
        # these are for the calculation of the dimensions
        names_weights_copy, indexes = get_inner_loop_parameter_dict(anil.learner.named_parameters())
        scheduler = Scheduler(len(names_weights_copy), args.buffer_size, grad_indexes=indexes).to(args.device)

        if args.resume !=0 and args.train:
            scheduler_file = '{0}/{2}/scheduler{1}'.format(args.logdir, args.test_epoch, exp_string)
            print(scheduler_file)
            scheduler.load_state_dict(torch.load(scheduler_file))
        scheduler_optimizer = torch.optim.Adam(scheduler.parameters(), lr=args.scheduler_lr)

    if args.t_decay: T = 1
    else: T = 0
    moving_avg_reward = 0
    lambda0 = args.lambda0

    if args.resume != 0 and args.train:
        T *= 0.999 ** args.test_epoch

    state_dict = copy.deepcopy(anil.state_dict())

    warmup = args.warmup

    for step, data in enumerate(dataloader):

        if step > args.metatrain_iterations:
            break

        task_losses = []
        task_losses_val = []
        task_acc = []
        task_accs_val = []

        if args.sampling_method is None:
            if args.datasource == 'miniimagenet' and not args.finetune:
                x_spt, y_spt, x_qry, y_qry, noisy_or_not = data
            else:
                x_spt, y_spt, x_qry, y_qry = data
                noisy_or_not = None
            x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                         x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
            for meta_batch in range(args.meta_batch_size):
                meta_batch_result = anil(x_spt[meta_batch], y_spt[meta_batch], x_qry[meta_batch],
                                                  y_qry[meta_batch])

                loss_val, acc_val = meta_batch_result

                task_losses.append(loss_val)
                task_acc.append(acc_val)

            meta_batch_loss_raw = torch.stack(task_losses)
            meta_batch_loss = meta_batch_loss_raw.mean()

            meta_batch_acc = torch.stack(task_acc).mean()

            optimiser.zero_grad()
            meta_batch_loss.backward()
            optimiser.step()

            train_steps.append(train_steps[-1] + args.meta_batch_size)
            train_losses.append(meta_batch_loss.item())
            train_accs.append(meta_batch_acc.item())
            if noisy_or_not is not None:
                clean_idx = torch.where(noisy_or_not == 0)
                noisy_idx = torch.where(noisy_or_not > 0)
                if len(clean_idx[0]) > 0:
                    train_clean_losses.append(meta_batch_loss_raw[clean_idx].mean().item())
                else:
                    train_clean_losses.append(-1)

                if len(noisy_idx[0]) > 0:
                    train_noisy_losses.append(meta_batch_loss_raw[noisy_idx].mean().item())
                else:
                    train_noisy_losses.append(-1)

        elif args.sampling_method == 'ATS':
            (x_spt, y_spt, x_qry, y_qry, x_spt_val, y_spt_val, x_qry_val, y_qry_val, noisy_or_not) = data

            scheduler.tasks = [[x1, y1, x2, y2] for (x1, y1, x2, y2) in zip(x_spt, y_spt, x_qry, y_qry)]

            names_weights_copy, _ = get_inner_loop_parameter_dict(anil.learner.named_parameters())

            pt = int(step / (args.metatrain_iterations + 1) * 100)

            meta_batch_loss_raw, _, all_task_weight = scheduler.get_weight(anil, pt)
            torch.cuda.empty_cache()
            all_task_prob = torch.softmax(all_task_weight.reshape(-1), dim=-1)

            selected_tasks_idx = scheduler.sample_task(all_task_prob, args.meta_batch_size, replace=args.replace)

            selected_losses = scheduler.compute_loss(selected_tasks_idx, anil)

            meta_batch_loss = torch.mean(selected_losses)

            # Prepare for second time forward
            fast_weights = OrderedDict(anil.learner.named_parameters())
            gradients = torch.autograd.grad(meta_batch_loss, fast_weights.values(), create_graph=True)

            fast_weights = OrderedDict(
                (name, param - anil.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

            # Second time forward, on validation dataset compute the gradient of scheduler to update it.
            x_spt_val, y_spt_val, x_qry_val, y_qry_val = x_spt_val.cuda(), y_spt_val.cuda(), \
                                                         x_qry_val.cuda(), y_qry_val.cuda()

            for meta_batch in range(args.meta_batch_size):
                x1, y1, x2, y2 = x_spt_val[meta_batch], y_spt_val[meta_batch], x_qry_val[meta_batch], y_qry_val[
                    meta_batch]
                x1, y1, x2, y2 = x1.squeeze(0).cuda(), y1.squeeze(0).cuda(), \
                                 x2.squeeze(0).cuda(), y2.squeeze(0).cuda()
                loss_train_val, acc_train_val = anil.forward_val(x1, y1, x2, y2, fast_weights)

                task_losses_val.append(loss_train_val.detach())
                task_accs_val.append(acc_train_val)

            loss = 0
            for i in selected_tasks_idx:
                loss = loss - scheduler.m.log_prob(i)
            reward = torch.stack(task_accs_val).mean()
            loss *= (reward - moving_avg_reward)

            moving_avg_reward = update_moving_avg(moving_avg_reward, reward, step)

            scheduler_optimizer.zero_grad()
            loss.backward()
            scheduler_optimizer.step()

            meta_batch_acc = 0
            task_losses = []
            task_acc = []

            # Third time forward, on training dataset, update the learner
            if step < warmup:
                optimiser.zero_grad()
                meta_batch_loss.backward()
                optimiser.step()
                anil.load_state_dict(state_dict)
            else:
                task_losses = []
                task_acc = []
                T *= 0.99
                meta_batch_loss_raw, _, all_task_weight = scheduler.get_weight(anil, pt, detach=True)
                all_task_prob = torch.softmax(all_task_weight.reshape(-1) / max(args.temperature, T), dim=-1)
                selected_tasks_idx = scheduler.sample_task(all_task_prob, args.meta_batch_size, replace=args.replace)
                selected_tasks_idx = torch.stack(selected_tasks_idx)
                selected_tasks_idx_unique, count = torch.unique(selected_tasks_idx, return_counts=True)

                for i, idx in enumerate(selected_tasks_idx_unique):
                    x1, y1, x2, y2 = scheduler.tasks[idx]
                    x1, y1, x2, y2 = x1.squeeze(0).cuda(), y1.squeeze(0).cuda(), \
                                     x2.squeeze(0).cuda(), y2.squeeze(0).cuda()
                    loss_train_train, acc_train_train = anil(x1, y1, x2, y2)

                    task_losses.extend([loss_train_train] * count[i])
                    task_acc.extend([acc_train_train] * count[i])

                meta_batch_loss = torch.stack(task_losses).mean()
                meta_batch_acc = torch.stack(task_acc).mean()

                optimiser.zero_grad()
                meta_batch_loss.backward()
                optimiser.step()

                train_steps.append(train_steps[-1] + 1)
                train_losses.append(meta_batch_loss_raw.data.tolist())
                train_probs.append(all_task_prob.data.tolist())
                # train_losses.append(meta_batch_loss.item())
                # train_accs.append(meta_batch_acc.item())
                clean_idx = torch.where(noisy_or_not == 0)
                noisy_idx = torch.where(noisy_or_not > 0)
                if len(clean_idx[0]) > 0:
                    train_clean_probs.append(all_task_prob[clean_idx].mean().item())
                    train_clean_losses.append(meta_batch_loss_raw[clean_idx].mean().item())
                else:
                    train_clean_probs.append(-1)
                    train_clean_losses.append(-1)

                if len(noisy_idx[0]) > 0:
                    train_noisy_probs.append(all_task_prob[noisy_idx].mean().item())
                    train_noisy_losses.append(meta_batch_loss_raw[noisy_idx].mean().item())
                else:
                    train_noisy_probs.append(-1)
                    train_noisy_losses.append(-1)

        else:
            print(f"Samplling method {args.sampling_method} not recognized.")
            raise NotImplementedError

        if step != 0 and step % Print_Iter == 0:
            if args.sampling_method == 'ATS':
                print(
                    'iter: {}, loss_all: {}, acc: {}, probs:{}'.format(
                    step, print_loss, print_acc, all_task_prob.data.tolist()[:3]))
                if step > warmup and not args.limit_data and args.noise > 0:
                    print("clean_prob_mean:", np.array(train_clean_probs[-Print_Iter:])[np.where(np.array(train_clean_probs[-Print_Iter:]) > 0)].mean(),
                          "noisy_prob_mean:", np.array(train_noisy_probs[-Print_Iter:])[np.where(np.array(train_noisy_probs[-Print_Iter:]) > 0)].mean(),
                          "clean_loss_mean:", np.array(train_clean_losses[-Print_Iter:])[np.where(np.array(train_clean_losses[-Print_Iter:]) > 0)].mean(),
                          "noisy_loss_mean:", np.array(train_noisy_losses[-Print_Iter:])[np.where(np.array(train_noisy_losses[-Print_Iter:]) > 0)].mean())

            else:
                print('iter: {}, loss_all: {}, acc: {}'.format(
                        step, print_loss, print_acc))
                if args.datasource == 'miniimagenet' and not args.finetune and not args.limit_data:
                    print("clean_loss_mean:", np.array(train_clean_losses[-Print_Iter:])[
                          np.where(np.array(train_clean_losses[-Print_Iter:]) > 0)].mean(),
                      "noisy_loss_mean:", np.array(train_noisy_losses[-Print_Iter:])[
                          np.where(np.array(train_noisy_losses[-Print_Iter:]) > 0)].mean())

            sys.stdout.flush()

            print_loss, print_acc, SPL_task_num = 0.0, 0.0, 0
        else:
            print_loss += meta_batch_loss / Print_Iter
            print_acc += meta_batch_acc / Print_Iter

        if step != 0 and step % Save_Iter == 0:
            test(args, anil, step)
            torch.save(anil.state_dict(),
                       '{0}/{2}/model{1}'.format(args.logdir, step, exp_string))
            if args.sampling_method == 'scheduler' or args.sampling_method == 'ALFA':
                torch.save(scheduler.state_dict(), '{0}/{2}/scheduler{1}'.format(args.logdir, step, exp_string))



def test(args, anil, test_epoch):
    res_acc = []
    meta_batch_size = args.meta_batch_size
    args.meta_batch_size = 1

    dataloader = MiniImagenet_MW(args, 'test')

    for step, (x_spt, y_spt, x_qry, y_qry) in enumerate(dataloader):
        if step > 600:
            break
        x_spt, y_spt, x_qry, y_qry = x_spt.squeeze(0).cuda(), y_spt.squeeze(0).cuda(), \
                                     x_qry.squeeze(0).cuda(), y_qry.squeeze(0).cuda()
        acc_val = anil(x_spt, y_spt, x_qry, y_qry)[1]
        res_acc.append(acc_val.item())

    res_acc = np.array(res_acc)

    print('test_epoch is {}, acc is {}, ci95 is {}'.format(test_epoch, np.mean(res_acc),
                                                                      1.96 * np.std(res_acc) / np.sqrt(
                                                                          args.num_test_task * args.meta_batch_size)))
    args.meta_batch_size = meta_batch_size


def main():
    # for miniimagenet
    final_layer_size = 800
    stride = 1

    if args.out_learner == 'anil':
        anil = ANIL(args, final_layer_size=final_layer_size, stride=stride).cuda()
        if args.resume != 0 and args.train == 1:
            model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
            print(model_file)
            anil.load_state_dict(torch.load(model_file))

        if args.noise > 0:
            anil.load_state_dict(torch.load(f"./model{args.pretrain_iter}_{args.update_batch_size}shot"))
            print(f"load model {args.pretrain_iter} successfully")

        meta_optimiser = torch.optim.Adam(list(anil.parameters()),
                                              lr=args.meta_lr, weight_decay=args.weight_decay)


    else:
        raise NotImplementedError


    if args.train == 1:
        with torch.backends.cudnn.flags(enabled=False):
            train(args, anil, meta_optimiser)

    else:
        for test_epoch in range(20000, 50000, 500):
            try:
                model_file = '{0}/{2}/model{1}'.format(args.logdir, test_epoch, exp_string)
                anil.load_state_dict(torch.load(model_file))
                test(args, anil, test_epoch)
            except IOError:
                continue


if __name__ == '__main__':
    main()