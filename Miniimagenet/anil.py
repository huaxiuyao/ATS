from collections import OrderedDict

import torch
import torch.nn as nn

from learner import Conv_Standard_ANIL


class ANIL(nn.Module):
    def __init__(self, args, final_layer_size, stride):
        super(ANIL, self).__init__()
        self.args = args
        self.learner = Conv_Standard_ANIL(args=args, x_dim=3, hid_dim=args.num_filters, z_dim=args.num_filters,
                                     final_layer_size=final_layer_size, stride=stride)
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_val = self.loss_fn
        if self.args.train:
            self.num_updates = self.args.num_updates
        else:
            self.num_updates = self.args.num_updates_test

    def forward(self, xs, ys, xq, yq):
        create_graph = True

        fast_weights = OrderedDict(self.learner.logits.named_parameters())

        for inner_batch in range(self.num_updates):
            logits = self.learner.functional_forward(xs, fast_weights, is_training=True)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)

            fast_weights = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights.items(), gradients)
            )

        query_logits = self.learner.functional_forward(xq, fast_weights)
        query_loss = self.loss_fn_val(query_logits, yq)

        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]

        return query_loss, query_acc

    def forward_val(self, xs, ys, xq, yq, fast_weights):
        create_graph = True

        fast_weights_logits = OrderedDict({"weight": fast_weights['logits.weight'], "bias": fast_weights['logits.bias']})

        for inner_batch in range(self.args.num_updates):
            logits = self.learner.functional_forward_val(xs, fast_weights, fast_weights_logits, is_training=True)
            loss = self.loss_fn(logits, ys)
            gradients = torch.autograd.grad(loss, fast_weights_logits.values(), create_graph=create_graph)

            fast_weights_logits = OrderedDict(
                (name, param - self.args.update_lr * grad)
                for ((name, param), grad) in zip(fast_weights_logits.items(), gradients)
            )

        query_logits = self.learner.functional_forward_val(xq, fast_weights, fast_weights_logits, is_training=True)
        query_loss = self.loss_fn(query_logits, yq)

        y_pred = query_logits.softmax(dim=1).max(dim=1)[1]
        query_acc = (y_pred == yq).sum().float() / yq.shape[0]
        return query_loss, query_acc

