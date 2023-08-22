import os
import torchvision
import torchvision.transforms as transforms
import torch
from tools.helpfunc import print_rank0
from torch.utils.data.distributed import DistributedSampler

def get_cifar10(data_path, network_config):
    print_rank0("loading CIFAR10")
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    batch_size = network_config['batch_size']
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]) 

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    train_sampler = DistributedSampler(trainset)
    trainloader = torch.utils.data.DataLoader(trainset, sampler = train_sampler,  batch_size=batch_size, shuffle=False, num_workers=8)
    

    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)
    test_sampler = DistributedSampler(testset)  # 创建一个分布式采样器
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=batch_size, shuffle=False, num_workers=8)

    return trainloader, testloader, train_sampler
