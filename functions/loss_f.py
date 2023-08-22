import torch
import torch.nn.functional as f


def psp(inputs, network_config):
    device = inputs.device
    shape = inputs.shape
    n_steps = network_config['n_steps']
    tau_s = network_config['tau_s']

    syn = torch.zeros(shape[0], shape[1], shape[2], shape[3]).to(device)
    syns = torch.zeros(shape[0], shape[1], shape[2], shape[3], n_steps).to(device)

    for t in range(n_steps):
        syn = syn - syn / tau_s + inputs[..., t]
        syns[..., t] = syn / tau_s

    return syns


class SpikeLoss(torch.nn.Module):
    """
    This class defines different spike based loss modules that can be used to optimize the SNN.
    """
    def __init__(self, network_config):
        super(SpikeLoss, self).__init__()
        self.network_config = network_config
        self.criterion = torch.nn.CrossEntropyLoss()

    def spike_kernel(self, outputs, target, network_config):
        delta = loss_kernel.apply(outputs, target, network_config)
        return 1 / 2 * torch.sum(delta ** 2)

class loss_kernel(torch.autograd.Function):  # a and u is the incremnet of each time steps
    @staticmethod
    def forward(ctx, outputs, target, network_config):
        # out_psp = psp(outputs, network_config)
        target_psp = psp(target, network_config)
        delta = outputs - target_psp
        return delta

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None

