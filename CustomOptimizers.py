import math

import torch
from torch.optim.optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, parameters, lr_min, lr_max, period, weight_decay=0, momentum=0, dampening = 0):
        """
        :param parameters: iterable of parameters to optimize or dicts defining
            parameter groups
        :param lr_min: lower bound for cosine rate decay (crd)
        :param lr_max: upper bound for crd
        :param iters: the number of iterations (used in crd)
        :param weight_decay: weight decay (weights in L-2 norm)
        """
        if lr_min < 0.0 or lr_max < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr_min, lr_max))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: " + weight_decay)
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: " + momentum)
        if dampening < 0:
            raise ValueError("Invalid dampening value: " + dampening)
        # question:
        defaults = {"lr_min": lr_min, "lr_max": lr_max, "period": period, "weight_decay": weight_decay,
                    "momentum": momentum, "dampening": dampening}
        super(SGD, self).__init__(parameters, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)

    def step(self, curr_iter, closure = None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:

            momentum_list = []
            param_list = []
            d_p_list = []

            for i, p in enumerate(group['params']):

                d_p_list.append(p.grad)
                param_list.append(p)
                # question:
                state = self.state[p]

                if 'prev_momentum' not in state:
                    momentum_list.append(None)
                else:
                    momentum_list.append(state['prev_momentum'])
                curr_lr = self.rateDecay(group['lr_min'], group['lr_max'], curr_iter, group['period'])
                d_p = d_p_list[i]

                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, group['weight_decay'])

                if group['momentum'] != 0:
                    momentum_grad = momentum_list[i]

                    if momentum_grad is None:
                        momentum_grad = torch.clone(d_p).detach()
                        momentum_list[i] = momentum_grad
                    else:
                        momentum_grad.mul_(momentum).add_(d_p, alpha=1 - group['dampening'])

                    d_p = momentum_grad
                print('before ' + str(param_list[i]))
                lr = self.rateDecay(group['lr_min'], group['lr_max'], curr_iter, group['period'])
                param_list[i].data.add_(d_p, alpha= -lr)
                for p, momentum in zip(param_list, momentum_list):
                    state = self.state[p]
                    state['prev_momentum'] = momentum
            print(param_list[0])
        return loss

    def rateDecay(self, lr_min, lr_max, iter, period):
        new_lr = lr_min + 1/2*(lr_max - lr_min)*(1 + math.cos(iter/period))
        return new_lr




