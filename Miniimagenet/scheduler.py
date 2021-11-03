from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
from torch.distributions.categorical import Categorical


class Scheduler(nn.Module):
    def __init__(self, N, buffer_size, grad_indexes, use_deepsets=True):
        super(Scheduler, self).__init__()
        self.percent_emb = nn.Embedding(100, 5)

        self.grad_lstm = nn.LSTM(N, 10, 1, bidirectional=True)
        self.loss_lstm = nn.LSTM(1, 10, 1, bidirectional=True)
        self.grad_lstm_2 = nn.LSTM(N, 10, 1, bidirectional=True)
        self.buffer_size = buffer_size
        self.grad_indexes = grad_indexes
        self.use_deepsets = use_deepsets
        self.cosine = torch.nn.CosineSimilarity(dim=-1, eps=1e-8)
        input_dim = 65
        if use_deepsets:
            self.h = nn.Sequential(nn.Linear(input_dim, 20), nn.Tanh(), nn.Linear(20, 10))
            self.fc1 = nn.Linear(input_dim + 10, 20)
        else:
            self.fc1 = nn.Linear(input_dim, 20)
        self.fc2 = nn.Linear(20, 1)
        self.tasks = []

    def forward(self, l, input, pt):
        x_percent = self.percent_emb(pt)

        grad_output_1, (hn, cn) = self.grad_lstm(input[0].reshape(1, len(input[0]), -1))
        grad_output_1 = grad_output_1.sum(0)
        grad_output_2, (hn, cn) = self.grad_lstm_2(input[1].reshape(1, len(input[1]), -1))
        grad_output_2 = grad_output_2.sum(0)
        grad_output = torch.cat((grad_output_1, grad_output_2), dim=1)

        loss_output, (hn, cn) = self.loss_lstm(l.reshape(1, len(l), 1))
        loss_output = loss_output.sum(0)
        x = torch.cat((x_percent, grad_output, loss_output), dim=1)

        if self.use_deepsets:
            x_C = (torch.sum(x, dim=1).unsqueeze(1) - x) / (len(x) - 1)
            x_C_mapping = self.h(x_C)
            x = torch.cat((x, x_C_mapping), dim=1)
            z = torch.tanh(self.fc1(x))
        else:
            z = torch.tanh(self.fc1(x))
        z = self.fc2(z)
        return z

    def sample_task(self, prob, size, replace=True):
        self.m = Categorical(prob)
        p = prob.detach().cpu().numpy()
        if len(np.where(p > 0)[0]) < size:
            print("exceptionally all actions")
            actions = torch.tensor(np.where(p > 0)[0])
        else:
            actions = np.random.choice(np.arange(len(prob)), p=p / np.sum(p), size=size,
                                       replace=replace)
            actions = [torch.tensor(x).cuda() for x in actions]
        return actions

    def compute_loss(self, selected_tasks_idx, maml):
        task_losses = []
        for task_idx in selected_tasks_idx:
            x1, y1, x2, y2 = self.tasks[task_idx]
            x1, y1, x2, y2 = x1.squeeze(0).cuda(), y1.squeeze(0).cuda(), \
                             x2.squeeze(0).cuda(), y2.squeeze(0).cuda()
            loss_val, acc_val = maml(x1, y1, x2, y2)
            task_losses.append(loss_val)
        return torch.stack(task_losses)

    def get_weight(self, maml, pt, per_step_task_embedding=None, detach=False, task_losses = None, return_grad=False):
        task_acc = []
        task_losses_new = []
        input_embedding_norm = []
        input_embedding_cos = []
        query_losses = []
        for (x1, y1, x2, y2) in self.tasks:
            # First time forward, on training dataset compute theta_0 and the grad to get the weight output from mentornet
            x1, y1, x2, y2 = x1.squeeze(0).cuda(), y1.squeeze(0).cuda(), \
                             x2.squeeze(0).cuda(), y2.squeeze(0).cuda()

            if task_losses is None:
                loss_val, acc_val = maml(x1, y1, x2, y2)
                task_losses_new.append(loss_val.detach())

            loss_support = maml.loss_fn(maml.learner(x1), y1)
            loss_query = maml.loss_fn(maml.learner(x2), y2)

            if return_grad: query_losses.append(loss_query.item())

            fast_weights = OrderedDict(maml.learner.named_parameters())
            task_grad_support = torch.autograd.grad(loss_support, fast_weights.values(), create_graph=False)
            task_grad_query = torch.autograd.grad(loss_query, fast_weights.values(), create_graph=False)
            task_grad = task_grad_query + task_grad_support

            task_layer_wise_grad_cos = []
            task_layer_wise_grad_norm = []
            for i in range(len(task_grad)):
                if i in self.grad_indexes:
                    task_layer_wise_grad_cos.append(self.cosine(task_grad_support[i].flatten().unsqueeze(0), task_grad_query[i].flatten().unsqueeze(0)))
                    task_layer_wise_grad_norm.append(task_grad[i].norm())

            del fast_weights
            del task_grad_support
            del task_grad_query
            del task_grad

            task_layer_wise_grad_norm = torch.stack(task_layer_wise_grad_norm)
            task_layer_wise_grad_cos = torch.stack(task_layer_wise_grad_cos)
            input_embedding_norm.append(task_layer_wise_grad_norm.detach())
            input_embedding_cos.append(task_layer_wise_grad_cos.detach())

        if task_losses is None:
            task_losses = torch.stack(task_losses_new)

        task_layer_inputs = [torch.stack(input_embedding_norm).cuda(), torch.stack(input_embedding_cos).cuda()]

        weight = self.forward(task_losses, task_layer_inputs,
                              torch.tensor([pt]).long().repeat(len(task_losses)).cuda())
        if detach:
            weight = weight.detach()

        if return_grad:
            return task_losses, task_acc, weight, task_layer_inputs, query_losses
        else:
            return task_losses, task_acc, weight