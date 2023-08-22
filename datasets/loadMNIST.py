import os
import torchvision.datasets
import torchvision.transforms as transforms
import torch
from torch.utils.data.distributed import DistributedSampler
from tools.helpfunc import print_rank0

def get_mnist(data_path, network_config):
    print_rank0("loading MNIST")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    batch_size = network_config['batch_size']

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(data_path, train=True, transform=transform_train, download=True)
    train_sampler = DistributedSampler(trainset)
    testset = torchvision.datasets.MNIST(data_path, train=False, transform=transform_test, download=True)
    test_sampler = DistributedSampler(testset)  # 创建一个分布式采样器

    trainloader = torch.utils.data.DataLoader(trainset, sampler = train_sampler,  batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, sampler=test_sampler, batch_size=batch_size, shuffle=False, num_workers=4)

    return trainloader, testloader, train_sampler