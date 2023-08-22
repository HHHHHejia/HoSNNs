import math

import torch
import torch.nn as nn
import torch.nn.functional as f
import functions.tsslbp as tsslbp
from tools.helpfunc import print_rank0
import tools.global_v as glv

class LinearLayer(nn.Linear):
    def __init__(self, network_config, config, name, in_shape):
        # extract information for kernel and inChannels
        in_features = config['n_inputs']
        out_features = config['n_outputs']
        self.layer_config = config
        self.network_config = network_config
        self.name = name
        self.type = config['type']
        self.in_shape = in_shape
        self.out_shape = [out_features, 1, 1]
        self.in_spikes = None
        self.out_spikes = None

        if 'weight_scale' in config:
            weight_scale = config['weight_scale']
        else:
            weight_scale = 1

        if type(in_features) == int:
            n_inputs = in_features
        else:
            raise Exception('inFeatures should not be more than 1 dimesnion. It was: {}'.format(in_features.shape))
        if type(out_features) == int:
            n_outputs = out_features
        else:
            raise Exception('outFeatures should not be more than 1 dimesnion. It was: {}'.format(out_features.shape))

        super(LinearLayer, self).__init__(n_inputs, n_outputs, bias=False)

        #init
        nn.init.kaiming_normal_(self.weight)
        self.weight = torch.nn.Parameter(weight_scale * self.weight, requires_grad=True)
    
        #bn
        self.is_bn = network_config["is_bn"]
        self.nobn = (self.is_bn == False) or ((network_config['model'] == "ALIF") and (self.name == "output"))
        if(self.nobn == False):
            self.bn = nn.BatchNorm1d(num_features=self.out_features)  # assuming self.out_features is the output feature number of your linear layer

        #deal alif thing
        if((network_config['model'] == "ALIF") and (self.name!= "output")):
            print_rank0("alif layer, using target train as init" )
            self.theta_v = nn.Parameter(torch.full(self.out_shape, 1.0/network_config['tau_v'], dtype=torch.float32), requires_grad=True)
            if(network_config['ckpt_v']):            
                print_rank0("load ckpt_v", network_config['ckpt_v'] )
                target_train = torch.load(network_config['ckpt_v'], map_location = glv.device)
                self.target_train = nn.Parameter(target_train[self.name], requires_grad=False)
            else:
                print_rank0("random ckpt_v" )
                self.target_train = nn.Parameter(torch.full((*self.out_shape, network_config['n_steps']), 0.5, dtype=torch.float32), requires_grad=False)
        else:
            print_rank0("lif layer, no theta_v or target train" )
            self.theta_v = None
            self.target_train =None

        print_rank0(self.name)
        print_rank0(self.in_shape)
        print_rank0(self.out_shape)
        print_rank0(list(self.weight.shape))
        print_rank0("-----------------------------------------")

    def forward(self, x):
        """
        """
        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3], x.shape[4])
        x = x.transpose(1, 2)
        y = f.linear(x, self.weight, self.bias)
        y = y.transpose(1, 2)

        if(self.nobn == False):
            y = self.bn(y)  # apply BatchNorm layer

        y = y.view(y.shape[0], y.shape[1], 1, 1, y.shape[2])
        return y

    def forward_pass(self, x):
        y = self.forward(x)
        y, out_spike= tsslbp.TSSLBP.apply(y, self.network_config, self.layer_config, self.name, self.target_train, self.theta_v)
        return y, out_spike

    def get_parameters(self):
        return self.weight

    def weight_clipper(self):
        w = self.weight.data
        w = w.clamp(-4, 4)
        self.weight.data = w

        if(self.theta_v!= None):
            v = self.theta_v.data
            v = v.clamp(min = 0)
            self.theta_v.data = v