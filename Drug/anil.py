from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


class ANIL(nn.Module):
    def __init__(self, args, learner):
        super(ANIL, self).__init__()
        self.args = args
        self.learner = learner
        self.loss_fn = nn.MSELoss()

    def filter(self, xs, ys, xq, yq):
        if xs.shape[0] > 2000:
            sel_id = np.random.choice(np.arange(xs.shape[0]), 2000, replace=False)
            xs = xs[sel_id]
            ys = ys[sel_id]

        if xq.shape[0] > 2000:
            sel_id = np.random.choice(np.arange(xq.shape[0]), 2000, replace=False)
            xq = xq[sel_id]
            yq = yq[sel_id]

        return xs, ys, xq, yq


    def forward(self, xs, ys, xq, yq):

        xs, ys, xq, yq = self.filter(xs, ys, xq, yq)
        create_graph = True

        fast_weights = OrderedDict(self.learner.logits.named_parameters())

        for inner_batch in range(self.args.num_updates):
            logits = self.learner.functional_forward(xs, fast_weights, is_training=True).squeeze()
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        query_logits = self.learner.functional_forward(xq, fast_weights, is_training=True).squeeze()
        query_loss = self.loss_fn(query_logits, yq)

        if self.args.train:
            return query_loss
        else:
            return query_loss, query_logits, yq

    def forward_val(self, xs, ys, xq, yq, fast_weights):
        xs, ys, xq, yq = self.filter(xs, ys, xq, yq)
        create_graph = True

        fast_weights_logits = OrderedDict({"weight": fast_weights['logits.weight'], "bias": fast_weights['logits.bias']})

        for inner_batch in range(self.args.num_updates):
            logits = self.learner.functional_forward_val(xs, fast_weights, fast_weights_logits, is_training=True).squeeze()
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights_logits.values(), create_graph=create_graph)

            fast_weights_logits = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights_logits.items(), gradients)
            )

        query_logits = self.learner.functional_forward_val(xq, fast_weights, fast_weights_logits, is_training=True).squeeze()
        query_loss = self.loss_fn(query_logits, yq)


        if self.args.train:
            return query_loss
        else:
            return query_loss, query_logits, yq