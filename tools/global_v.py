import torch
import configparser

max_accuracy = 0
min_loss = 1000
best_acc = 0
best_epoch = 0
    
n_steps = None
syn_a = None
tau_s = None


#read attack config 
config = configparser.ConfigParser()
config.read('./config/mnist.ini')
device = torch.device("cuda")


def init(n_t, ts):   
    global n_steps, syn_a, partial_a, tau_s
    n_steps = n_t
    tau_s = ts
    syn_a = torch.zeros(1, 1, 1, 1, n_steps).to(device)
    syn_a[..., 0] = 1
    for t in range(n_steps-1):
        syn_a[..., t+1] = syn_a[..., t] - syn_a[..., t] / tau_s 
    syn_a /= tau_s
    
    partial_a = torch.zeros((1, 1, 1, 1, n_steps, n_steps)).to(device)
    for t in range(n_steps):
        if t > 0:
            partial_a[..., t] = partial_a[..., t - 1] - partial_a[..., t - 1] / tau_s 
        partial_a[..., t, t] = 1/tau_s