'''
This module is based on Liangqu Long's code:
https://github.com/dragen1860/MAML-Pytorch/blob/master/learner.py
'''


import  torch
from    torch import nn
from    torch.nn import functional as F
import  numpy as np


def update_weights(loss, vars, lr, first_order=False, grad_clip=None):
    if first_order:
        grad = torch.autograd.grad(loss, vars)
    else:
        grad = torch.autograd.grad(loss, vars, retain_graph=True, create_graph=True)

    if grad_clip is not None:
        grad_norm = torch.norm(torch.stack([torch.norm(g.detach())  # .to(device)
                                            for g in grad]))
        if grad_norm > grad_clip:
            grad = [g / grad_norm * grad_clip for g in grad]

    return [v - lr * g for v, g in zip(vars, grad)]

class Learner(nn.Module):
    """

    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        self.current_vars = None
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear_homo':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                stdv = 1. / np.sqrt(param[1])
                w.data.uniform_(-stdv, stdv)
                self.vars.append(w)

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                b = nn.Parameter(torch.zeros(param[0]))
                # gain=1 according to cbfinn's implementation
                # torch.nn.init.kaiming_normal_(w)
                # self.vars.append(w)
                # # [ch_out]
                # self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # [June 2022] Zero-initialization of the bias and normal
                #  initialization of the weights led to poor
                #  performance on the Sine benchmark. Using uniform
                #  initialization instead as is default in pytorch.
                stdv = 1. / np.sqrt(param[1])
                w.data.uniform_(-stdv, stdv)
                b.data.uniform_(-stdv, stdv)
                self.vars.append(w)
                self.vars.append(b)

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'cond_max_pool2d', 'flatten', 'reshape', 'leakyrelu',
                          'sigmoid', 'log_softmax']:
                continue
            else:
                raise NotImplementedError


    def reset_model(self):
        self.current_vars = None

    def update_current_weights(self, loss, lr, first_order=False, grad_clip=None,
                               only_last=0):
        vars = self.current_vars
        if vars is None:
            vars = self.parameters()

        out_vars = []
        if only_last > 0:
            out_vars = [vars[i] for i in range(len(vars)-only_last)]
            vars = vars[-only_last:]
        updated_vars = update_weights(loss, vars, lr, first_order, grad_clip)
        out_vars.extend(updated_vars)

        self.current_vars = out_vars


    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear' or name == 'linear_homo':
                tmp = f'{name}:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'cond_max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d, cond:%d)'%(
                    param[0], param[1], param[2], param[3])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits',
                          'bn', 'log_softmax']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True, last_layer_input=None):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars if self.current_vars is None else self.current_vars

        idx = 0
        bn_idx = 0

        for layer, (name, param) in enumerate(self.config):
            if last_layer_input is not None and layer==len(self.config)-1:
                last_layer_input = last_layer_input.reshape(1,-1).repeat(x.shape[0],1)
                x = torch.cat((x, last_layer_input), dim=1)

            if name == 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name == 'linear_homo':
                w = vars[idx]
                x = F.linear(x, w)
                idx += 1
                # print('forward:', idx, x.norm().item())
            elif name == 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name == 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name == 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name == 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name == 'relu':
                x = F.relu(x, inplace=param[0])
            elif name == 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name == 'tanh':
                x = torch.tanh(x)
            elif name == 'sigmoid':
                x = torch.sigmoid(x)
            elif name == 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name == 'cond_max_pool2d':
                if x.shape[-1] > param[3]:
                    x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name == 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])
            elif name == 'log_softmax':
                x = torch.log_softmax(x, dim=param[0])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars