import math

import torch
from torch.optim.optimizer import Optimizer

class CosineRateDecay(Optimizer):
    def __init__(self, parameters, lr_min, lr_max, epochs, weight_decay=0, momentum=0, dampening = 0):
        """
        :param parameters: iterable of parameters to optimize or dicts defining
            parameter groups
        :param lr: learning rate (required)
        """
        if lr_min < 0.0 or lr_max < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr_min, lr_max))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        defaults = {"lr_min": lr_min, "lr_max": lr_max, "epochs": epochs, "weight_decay": weight_decay,
                    "momentum": momentum, "dampening": dampening}
        super(CosineRateDecay, self).__init__(parameters, defaults)

    def __setstate__(self, state):
        super(CosineRateDecay, self).__setstate__(state)

    def step(self, curr_epoch, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            curr_lr = self.rateDecay(group['lr_min'], group['lr_max'],
                                     curr_epoch, group['epochs'])
            momentum_list = []
            params_with_grad = []
            d_p_list = []

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad is not None:
                    d_p_list.append(p.grad)
                    params_with_grad.append(p)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_list.append(None)
                    else:
                        momentum_list.append(state['momentum_buffer'])
                curr_lr = self.rateDecay(group['lr_min'], group['lr_max'], curr_epoch, group['epochs'])
                self.crd(params_with_grad, d_p_list, momentum_list, group['weight_decay'], group['momentum'], curr_lr,
                         group['dampening'])

                for p, momentum in zip(params_with_grad, momentum_list):
                    state = self.state[p]
                    state['momentum_buffer'] = momentum
        return loss

    def rateDecay(self, lr_min, lr_max, epoch, period):
        new_lr = lr_min + 1/2*(lr_max - lr_min)*(1 + math.cos(epoch/period))
        return new_lr

    def crd(self, params, d_p_list, momentum_list, weight_decay, momentum, lr, dampening):
        for i, param in enumerate(params):
            d_p = d_p_list[i]

            if weight_decay != 0:
                d_p = d_p.add(param, weight_decay)

            if momentum != 0:
                momentum_grad = momentum_list[i]

                if momentum_grad is None:
                    momentum_grad = torch.clone(d_p).detach()
                    momentum_list[i] = momentum_grad
                else:
                    momentum_grad.mul_(momentum).add_(d_p, alpha=1 - dampening)

                d_p = momentum_grad

            params[i].add_(d_p, alpha = -lr)




