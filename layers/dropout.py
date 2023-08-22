import torch.nn as nn
import torch.nn.functional as f
from tools.helpfunc import print_rank0


class DropoutLayer(nn.Dropout):
    def __init__(self, config, name, inplace=False):
        self.name = name
        self.type = config['type']
        if 'p' in config:
            p = config['p']
        else:
            p = 0.5
        super(DropoutLayer, self).__init__(p, inplace)
        print_rank0(self.name)
        print_rank0("p: %.2f" % p)
        print_rank0("-----------------------------------------")

    def forward(self, x):
        result = f.dropout(x.reshape((x.shape[0], x.shape[1] * x.shape[2], x.shape[3], 1, 1, x.shape[4])),
                             self.p, self.training, self.inplace)
        return result.reshape((result.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]))

    def weight_clipper(self):
        return
