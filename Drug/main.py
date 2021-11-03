import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import random
from scipy.stats import pearsonr

from anil import ANIL
from data import MetaLearningSystemDataLoader
from learner import FCNet
from scheduler import Scheduler

parser = argparse.ArgumentParser(description='graph transfer')
parser.add_argument('--datasource', default='drug', type=str,
                    help='drug')
parser.add_argument('--dim_w', default=1024, type=int, help='dimension of w')
parser.add_argument('--dim_y', default=1, type=int, help='dimension of w')
parser.add_argument('--dataset_name', default='assay', type=str,
                    help='dataset_name.')
parser.add_argument('--dataset_path', default='ci9b00375_si_002.txt', type=str,
                    help='dataset_path.')
parser.add_argument('--type_filename', default='ci9b00375_si_001.txt', type=str,
                    help='type_filename.')
parser.add_argument('--compound_filename', default='ci9b00375_si_003.txt', type=str,
                    help='Directory of data files.')

parser.add_argument('--fp_filename', default='compound_fp.npy', type=str,
                    help='fp_filename.')

parser.add_argument('--target_assay_list', default='591252', type=str,
                    help='target_assay_list')

parser.add_argument('--train_seed', default=0, type=int, help='train_seed')

parser.add_argument('--val_seed', default=0, type=int, help='val_seed')

parser.add_argument('--test_seed', default=0, type=int, help='test_seed')

parser.add_argument('--train_val_split', default=[0.9588, 0.0177736202, 0.023386342], type=list, help='train_val_split')
parser.add_argument('--num_evaluation_tasks', default=100, type=int, help='num_evaluation_tasks')
parser.add_argument('--drug_group', default=17, type=int, help='drug group')
parser.add_argument('--metatrain_iterations', default=20, type=int,
                    help='number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
parser.add_argument('--meta_batch_size', default=8, type=int, help='number of tasks sampled per meta-update')
parser.add_argument('--update_lr', default=0.01, type=float, help='inner learning rate')
parser.add_argument('--meta_lr', default=0.001, type=float, help='the base learning rate of the generator')
parser.add_argument('--num_updates', default=5, type=int, help='num_updates in anil')
parser.add_argument('--test_num_updates', default=5, type=int, help='num_updates in anil')
parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
parser.add_argument('--update_batch_size', default=1, type=int,
                    help='number of examples used for inner gradient update (K for K-shot learning).')
parser.add_argument('--update_batch_size_eval', default=15, type=int,
                    help='number of examples used for inner gradient test (K for K-shot learning).')
## Logging, saving, and testing options
parser.add_argument('--logdir', default='./logs', type=str,
                    help='directory for summaries and checkpoints.')
parser.add_argument('--data_dir', default='./', type=str, help='directory for datasets.')
parser.add_argument('--resume', default=0, type=int, help='resume training if there is a model available')
parser.add_argument('--train', default=1, type=int, help='True to train, False to test.')
parser.add_argument('--test_epoch', default=-1, type=int, help='test epoch, only work when test start')
parser.add_argument('--sampling_method', default='ATS', help='specified sampling method')
parser.add_argument('--trial', default=0, type=int, help='trial for each layer')
parser.add_argument('--debug', action='store_true', default=False, help='debug mode')
parser.add_argument('--buffer_size', type=int, default=20)
parser.add_argument("--device", default='cuda')
parser.add_argument("--noise", default=0, type=float)
parser.add_argument("--gamma", default=0.25, type=float)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument('--simple_loss', default=False, action='store_true')

args = parser.parse_args()
print(args)

random.seed(1)
np.random.seed(2)

exp_string = 'data_' + str(args.datasource) + '.mbs_' + str(
    args.meta_batch_size) + '.metalr' + str(args.meta_lr) + '.innerlr' + str(args.update_lr)
if args.trial > 0:
    exp_string += '.trial{}'.format(args.trial)
if args.sampling_method is not None:
    exp_string += '.Sample-{}'.format(args.sampling_method)
if args.noise > 0:
    exp_string += f'_noise_{args.noise}'
if args.simple_loss:
    exp_string += '_simple_loss'


exp_string += '.drug_group-{}'.format(args.drug_group)

print(exp_string)
if not os.path.exists(args.logdir + '/' + exp_string + '/'):
    os.makedirs(args.logdir + '/' + exp_string + '/')

args.data_dir = args.data_dir+ '/' +args.datasource

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
        if "1.weight" in name or '1.bias' in name: continue
        if param.requires_grad:
            param_dict[name] = param.to(device=args.device)
            indexes.append(i)

    return param_dict, indexes


def update_moving_avg(mv, reward, count):
    return mv + (reward.item() - mv) / (count + 1)



def train(args, anil, optimiser, dataloader):
    Print_Iter = 100
    Save_Iter = 500
    print_loss, print_acc, print_loss_support = 0.0, 0.0, 0.0

    train_steps = [1]
    train_losses = []
    train_probs = []

    data_each_epoch = 4100
    moving_avg_reward = 0

    if args.debug:
        print("Training start:")
        sys.stdout.flush()

    if args.sampling_method == 'ATS':
        names_weights_copy, indexes = get_inner_loop_parameter_dict(anil.learner.named_parameters())
        scheduler = Scheduler(len(names_weights_copy), args.buffer_size, grad_indexes=indexes).to(args.device)
        scheduler_optimizer = torch.optim.Adam(scheduler.parameters(), lr=0.0001)

    count = 0

    for epoch in range(args.metatrain_iterations):

        if args.sampling_method == 'ATS':
            train_data_all = dataloader.get_train_val_batches()
        else:
            train_data_all = dataloader.get_train_batches()

        for step, cur_data in enumerate(train_data_all):
            task_losses = []
            task_losses_val = []

            if args.sampling_method is None:
                support_set_x, support_set_y, support_set_z, support_set_assay, \
                target_set_x, target_set_y, target_set_z, target_set_assay, seed = cur_data

                for meta_batch in range(args.meta_batch_size):
                    x1, y1, x2, y2 = support_set_x[meta_batch].squeeze().float().to(args.device), support_set_y[meta_batch].squeeze().float().to(args.device), \
                                    target_set_x[meta_batch].squeeze().float().to(args.device), target_set_y[meta_batch].squeeze().float().to(args.device)
                    loss_val = anil(x1, y1, x2, y2)
                    task_losses.append(loss_val)

                meta_batch_loss_raw = torch.stack(task_losses)
                meta_batch_loss = meta_batch_loss_raw.mean()

                optimiser.zero_grad()
                meta_batch_loss.backward()
                optimiser.step()

            elif args.sampling_method == 'ATS':

                support_set_x, support_set_y, support_set_z, support_set_assay, \
                target_set_x, target_set_y, target_set_z, target_set_assay, seed, \
                support_set_x_val, support_set_y_val, support_set_z_val, support_set_assay_val, \
                target_set_x_val, target_set_y_val, target_set_z_val, target_set_assay_val, seed_val = cur_data

                scheduler.add_new_tasks([[x1, y1, x2, y2] for (x1, y1, x2, y2) in
                                      zip(support_set_x, support_set_y, target_set_x, target_set_y)])
                names_weights_copy, _ = get_inner_loop_parameter_dict(anil.learner.named_parameters())
                pt = int(count / (args.metatrain_iterations * data_each_epoch) * 100)

                task_losses, _, all_task_weight = scheduler.get_weight(anil, pt)
                all_task_prob = torch.softmax(all_task_weight.reshape(-1), dim=-1)

                selected_tasks_idx = scheduler.sample_task(all_task_prob, args.meta_batch_size)

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
                for meta_batch in range(args.meta_batch_size):
                    x1, y1, x2, y2 = support_set_x_val[meta_batch].float().squeeze().to(args.device), \
                                    support_set_y_val[meta_batch].float().squeeze().to(args.device), \
                                    target_set_x_val[meta_batch].float().squeeze().to(args.device), \
                                    target_set_y_val[meta_batch].float().squeeze().to(args.device)
                    loss_train_val = anil.forward_val(x1, y1, x2, y2, fast_weights)

                    task_losses_val.append(loss_train_val)
                loss = 0
                for i in selected_tasks_idx:
                    loss = loss - scheduler.m.log_prob(i)
                loss *= (-torch.stack(task_losses_val).mean())

                # loss *= (reward - moving_avg_reward)
                # moving_avg_reward = update_moving_avg(moving_avg_reward, reward, step)

                scheduler_optimizer.zero_grad()
                loss.backward()
                scheduler_optimizer.step()

                # Third time forward, on training dataset, update the learner
                task_losses = []

                meta_batch_loss_raw, _, all_task_weight = scheduler.get_weight(anil, pt)
                all_task_prob = torch.softmax(all_task_weight.detach().reshape(-1), dim=-1)
                selected_tasks_idx = scheduler.sample_task(all_task_prob, args.meta_batch_size)
                selected_tasks_idx = torch.stack(selected_tasks_idx)

                selected_tasks_idx, task_count = torch.unique(selected_tasks_idx, return_counts=True)

                for i, idx in enumerate(selected_tasks_idx):
                    x1, y1, x2, y2 = scheduler.tasks[idx]
                    x1, y1, x2, y2 = x1.squeeze(0).float().to(args.device), y1.squeeze(0).float().to(args.device), \
                                     x2.squeeze(0).float().to(args.device), y2.squeeze(0).float().to(args.device)
                    loss_train_train = anil(x1, y1, x2, y2)
                    task_losses.extend([loss_train_train] * task_count[i])\

                meta_batch_loss = torch.stack(task_losses).mean()

                optimiser.zero_grad()
                meta_batch_loss.backward()
                optimiser.step()

                train_steps.append(train_steps[-1] + 1)
                train_losses.append(meta_batch_loss_raw.data.tolist())
                train_probs.append(all_task_prob.data.tolist())

            else:
                print(f"Samplling method {args.sampling_method} not recognized.")
                raise NotImplementedError


            if step != 0 and step % Print_Iter == 0:
                print('epoch: {}, iter: {}, mse: {}'.format(epoch, step, print_loss))
                sys.stdout.flush()
                print_loss = 0.0
            else:
                print_loss += meta_batch_loss / Print_Iter

            if step != 0 and step % Save_Iter == 0:
                torch.save(anil.learner.state_dict(),
                           '{0}/{2}/model{1}'.format(args.logdir, step+epoch*data_each_epoch, exp_string))
                if args.sampling_method == 'schedulernet' or args.sampling_method == 'ATS':
                    torch.save(scheduler.state_dict(), '{0}/{2}/schedulernet{1}'.format(args.logdir, step+epoch*data_each_epoch, exp_string))

            count += 1

def test(args, epoch, anil, dataloader):
    res_acc = []

    valid_cnt = 0

    test_data_all = dataloader.get_test_batches()

    for step, cur_data in enumerate(test_data_all):
        support_set_x, support_set_y, support_set_z, support_set_assay, \
        target_set_x, target_set_y, target_set_z, target_set_assay, seed = cur_data

        mse_loss, pred_label, actual_label = anil(support_set_x[0].squeeze().to(args.device), support_set_y[0].squeeze().to(args.device),
                                                  target_set_x[0].squeeze().to(args.device), target_set_y[0].squeeze().to(args.device))

        r2 = np.square(pearsonr(actual_label.cpu().numpy(), pred_label.detach().cpu().numpy())[0])
        res_acc.append(r2)

        if r2 > 0.3:
            valid_cnt += 1

    res_acc = np.array(res_acc)
    median = np.median(res_acc, 0)
    mean = np.mean(res_acc, 0)

    print('epoch is: {} mean is: {}, median is: {}, cnt>0.3 is: {}'.format(epoch, mean, median, valid_cnt))
    return mean


def main():
    learner = FCNet(args=args, x_dim=args.dim_w, hid_dim=500).to(args.device)

    anil = ANIL(args, learner)

    if args.resume == 1 and args.train == 1:
        model_file = '{0}/{2}/model{1}'.format(args.logdir, args.test_epoch, exp_string)
        print("model_file:", model_file)
        learner.load_state_dict(torch.load(model_file))

    meta_optimiser = torch.optim.Adam(list(learner.parameters()),
                                      lr=args.meta_lr, weight_decay=args.weight_decay)

    dataloader = MetaLearningSystemDataLoader(args, target_assay=args.target_assay_list)
    mean = []
    if args.train == 1:
        with torch.backends.cudnn.flags(enabled=False):
            train(args, anil, meta_optimiser, dataloader)
    else:
        args.meta_batch_size = 1
        for epoch in range(500, 80000, 100):
            model_file = '{0}/{2}/model{1}'.format(args.logdir, epoch, exp_string)
            if os.path.exists(model_file):
                learner.load_state_dict(torch.load(model_file))
                mean.append(test(args, epoch, anil, dataloader))
            else:
                continue

if __name__ == '__main__':
    main()
