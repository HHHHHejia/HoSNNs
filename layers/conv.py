import math
import torch
import torch.nn as nn
import torch.nn.functional as f
import functions.tsslbp as tsslbp
from tools.helpfunc import print_rank0
import tools.global_v as glv

class ConvLayer(nn.Conv3d):
    def __init__(self, network_config, config, name, in_shape, groups=1):
        self.name = name
        self.layer_config = config
        self.network_config = network_config
        self.type = config['type']
        in_features = config['in_channels']
        out_features = config['out_channels']
        kernel_size = config['kernel_size']

        if 'padding' in config:
            padding = config['padding']
        else:
            padding = 0

        if 'stride' in config:
            stride = config['stride']
        else:
            stride = 1

        if 'dilation' in config:
            dilation = config['dilation']
        else:
            dilation = 1

        if 'weight_scale' in config:
            weight_scale = config['weight_scale']
        else:
            weight_scale = 1

        # kernel
        if type(kernel_size) == int:
            kernel = (kernel_size, kernel_size, 1)
        elif len(kernel_size) == 2:
            kernel = (kernel_size[0], kernel_size[1], 1)
        else:
            raise Exception('kernelSize can only be of 1 or 2 dimension. It was: {}'.format(kernel_size.shape))

        # stride
        if type(stride) == int:
            stride = (stride, stride, 1)
        elif len(stride) == 2:
            stride = (stride[0], stride[1], 1)
        else:
            raise Exception('stride can be either int or tuple of size 2. It was: {}'.format(stride.shape))

        # padding
        if type(padding) == int:
            padding = (padding, padding, 0)
        elif len(padding) == 2:
            padding = (padding[0], padding[1], 0)
        else:
            raise Exception('padding can be either int or tuple of size 2. It was: {}'.format(padding.shape))

        # dilation
        if type(dilation) == int:
            dilation = (dilation, dilation, 1)
        elif len(dilation) == 2:
            dilation = (dilation[0], dilation[1], 1)
        else:
            raise Exception('dilation can be either int or tuple of size 2. It was: {}'.format(dilation.shape))

        #init para
        super(ConvLayer, self).__init__(in_features, out_features, kernel, stride, padding, dilation, groups,
                                        bias=False)
        self.weight = torch.nn.Parameter(weight_scale * self.weight, requires_grad=True)

        self.in_shape = in_shape
        self.out_shape = [out_features, int((in_shape[1]+2*padding[0]-kernel[0])/stride[0]+1),
                          int((in_shape[2]+2*padding[1]-kernel[1])/stride[1]+1)]

        self.is_bn = network_config["is_bn"]
        if(self.is_bn == True):
            self.bn = nn.BatchNorm3d(num_features=self.out_shape[0])  # assuming self.out_shape[0] is the output feature number of your conv layer

        #deal alif thing
        if(network_config['model'] == "ALIF"):
            print_rank0("alif net, using target train as init" )
            self.theta_v = nn.Parameter(torch.full(self.out_shape, 1.0/network_config['tau_v'], dtype=torch.float32), requires_grad=True)
            if(network_config['ckpt_v']):
                target_train = torch.load(network_config['ckpt_v'], map_location = glv.device)
                self.target_train = nn.Parameter(target_train[self.name], requires_grad=True)
            else:
                self.target_train = nn.Parameter(torch.rand(*self.out_shape, network_config['n_steps'], dtype=torch.float32), requires_grad=True)
        else:
            print_rank0("lif net, no theta_v or target train" )
            self.theta_v = None
            self.target_train =None

        print_rank0(self.name)
        print_rank0(self.in_shape)
        print_rank0(self.out_shape)
        print_rank0(list(self.weight.shape))
        print_rank0("-----------------------------------------")

    def forward(self, x):
        x = f.conv3d(x, self.weight, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)

        if(self.is_bn == True):
            x = self.bn(x)  # apply BatchNorm layer
        return x
    def get_parameters(self):
        return self.weight

    def forward_pass(self, x):
        y = self.forward(x)
        y, out_spike = tsslbp.TSSLBP.apply(y, self.network_config, self.layer_config,self.name, self.target_train, self.theta_v)
        return y, out_spike

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w

        if(self.theta_v!= None):
            v = self.theta_v.data
            v = v.clamp(min = 0)
            self.theta_v.data = v
