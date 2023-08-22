import torch
import torch.nn as nn
import layers.conv as conv
import layers.pooling as pooling
import layers.dropout as dropout
import layers.linear as linear
import functions.loss_f as f
import torch.distributed as dist


def print_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


class Network(nn.Module):
    def __init__(self, network_config, layers_config, input_shape):
        super(Network, self).__init__()
        self.network_config = network_config
        self.layers_config = layers_config

        self.layers = nn.ModuleList()

        print_rank0("Network Structure:")
        for key in layers_config:
            c = layers_config[key]
            if c['type'] == 'conv':
                layer = conv.ConvLayer(network_config, c, key, input_shape)
                self.layers.append(layer)
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'linear':
                layer = linear.LinearLayer(network_config, c, key, input_shape)
                self.layers.append(layer)
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'pooling':
                layer = pooling.PoolLayer(network_config, c, key, input_shape)
                self.layers.append(layer)
                input_shape = self.layers[-1].out_shape
            elif c['type'] == 'dropout':
                layer = dropout.DropoutLayer(c, key)
                self.layers.append(layer)
            else:
                raise Exception('Undefined layer type. It is: {}'.format(c['type']))

        print_rank0("-----------------------------------------")

    def forward(self, spike_input, is_train):
        spikes = f.psp(spike_input, self.network_config)
        
        # Initialize an empty dictionary to store the average spike train for each layer
        avg_spike_trains = {}

        for l in self.layers:
            if l.type == "dropout":
                if is_train:
                    spikes = l(spikes)
            elif l.type == "pooling":
                spikes = l.forward_pass(spikes)
            else:
                spikes, out_train = l.forward_pass(spikes)
                if(out_train!= None):
                    avg_spike_trains[l.name] = out_train.detach()

        return spikes, avg_spike_trains

    def get_parameters(self):
        return self.parameters()

    def weight_clipper(self):
        for l in self.layers:
            l.weight_clipper()

    def train(self, mode=True):
        super().train(mode)
        for l in self.layers:
            l.train(mode)

    def eval(self):
        for l in self.layers:
            l.eval()
