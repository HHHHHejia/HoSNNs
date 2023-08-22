import torch, argparse
import tools.global_v as glv
from tools.helpfunc import advtrain, advtest, init_distributed_mode, advtest
from tools.helpfunc import print_rank0, wandb_init_rank0
from tools.network_parser import parse
from datasets.loadCIFAR100 import get_cifar100
from tools.cnns import Network
from torch.nn.parallel import DistributedDataParallel
import os
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'

#set gpu
init_distributed_mode()
device =glv.device

#create parser to choose dataset
parser = argparse.ArgumentParser()
parser.add_argument('-d', type=int, help='2:cifar100')
args = parser.parse_args()

#read  config 
data_type = args.d
if data_type == 2:
    params = parse('Networks/CIFAR100.yaml')
else:
    print_rank0("Wrong dataset!")
    exit()

#net default para
dataset = params['Network']['dataset']
ckpt = params['DEFAULT']['ckpt']
mode = {}
mode['cleantrain'] = bool(params['DEFAULT']['cleantrain'])
mode['advtrain'] = bool(params['DEFAULT']['advtrain'])
mode['advtest'] =bool(params['DEFAULT']['advtest'])
print_rank0(mode)


#get dataset
data_path = params['Network']['data_path']
if dataset == 'CIFAR100':
    train_loader, test_loader, train_sampler, test_sampler= get_cifar100(data_path, params['Network'])
else:
    print_rank0("Wrong dataset")

glv.init(params['Network']['n_steps'], params['Network']['tau_s'] )


#get net 
input_shape = list(train_loader.dataset[0][0].shape)
net = Network(params['Network'], params['Layers'], input_shape).to(device)	
net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
net = DistributedDataParallel(net, device_ids=[device])


#load ckpt
if ckpt:
    print_rank0("using check point: ", ckpt)
    checkpoint = torch.load(ckpt, map_location = device)
    net.load_state_dict(checkpoint['net'])


#set wandb
run = wandb_init_rank0(project="aaai-final", config=params)


#start
print_rank0("using ",params['Network']["model"], " network!!!!")
#1.clean train
if mode['cleantrain']:   
    advtrain(net, train_loader, test_loader, train_sampler,
             params, "clean")
    
#2.adversarial training
if mode['advtrain']:
    advtrain(net, train_loader, test_loader,train_sampler,
              params, "adv")
    

#3.adv test
if mode['advtest']:
    advtest(net, test_loader, params, True, ckpt)

print_rank0("Best Accuracy: %.3f, at epoch: %d \n"%(glv.best_acc, glv.best_epoch))
