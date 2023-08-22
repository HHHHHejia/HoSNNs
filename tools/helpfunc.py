import torch, os, logging
import functions.loss_f as loss_f
import numpy as np
from torch.nn.utils import clip_grad_norm_
import tools.global_v as glv
from tools.global_v import device
from tools.attack import fast_gradient_method, projected_gradient_descent, bim_attack, r_fgsm
import torch.distributed as dist
import wandb
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

def print_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)

def wandb_init_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        wandb.init(*args, **kwargs)

def wandb_log_rank0(*args, **kwargs):
    if dist.get_rank() == 0:
        wandb.log(*args, **kwargs)

def init_distributed_mode():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
    else:
        print_rank0('Not using distributed mode')
        return

    torch.cuda.set_device(gpu)
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    assert torch.distributed.is_initialized()



def calculate_loss(outputs, labels, network_config, layers_config, err):
    loss_function = network_config['loss']

    if loss_function == "softmax":
        return err.spike_soft_max(outputs, labels)
    else:
        raise Exception('Unrecognized loss function.')

def save_ckpt(network, epoch, name) :
    state = {
        'net': network.state_dict(),
        'epoch': epoch,
    }
    name = './checkpoint/tmp/' + name + 'best.pth'
    torch.save(state, name)
    print_rank0("Saved to ", name)

def save_target(avg_mem_trains_acc, dataloader, mode, method, act_eps):
    #save spike
    # Calculate the average spike train for the whole dataset
    for l_name in avg_mem_trains_acc.keys():
        dist.all_reduce(avg_mem_trains_acc[l_name])
        avg_mem_trains_acc[l_name] /= (len(dataloader) * dist.get_world_size())

    if dist.get_rank() == 0:
        # Save the dictionary of average spike trains to a file
        name =  "./spikedata/tmp/" +"avg_mem_" + mode + "_" + method + "_"  + str(act_eps)[:4] +'.pt'
        torch.save(avg_mem_trains_acc, name)
        print_rank0("Saved target to ", name)

    return avg_mem_trains_acc

def train(network, trainloader, opti, epoch, network_config, layers_config, err, method = None, act_eps = None, tau_v= None):
    network.train()
    logging.info('\nEpoch: %d', epoch)
    train_loss = 0
    correct = 0
    total = 0
    n_steps = network_config['n_steps']

    avg_mem_trains_acc = {l.name: 0 for l in network.module.layers if l.type in ["conv", "linear"]}

    for _, (inputs, labels) in tqdm(enumerate(trainloader)):

        if len(inputs.shape) < 5:
            inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
        # forward pass
        labels = labels.to(device)
        inputs = inputs.to(device)
        inputs.type(torch.float32)

        #gen adv target
        if method == "clean":
            pass
        elif method == "fgm":
            inputs = fast_gradient_method(network, inputs, labels, act_eps, network_config)
        elif method == "pgd":
            inputs = projected_gradient_descent(network, inputs, labels, act_eps, network_config)

        # forward pass
        outputs, avg_mem_trains= network.forward(inputs, True)

        #Accumulate the average spike train for each 
        if(avg_mem_trains!= {}):
            for l_name in avg_mem_trains_acc.keys():
                avg_mem_trains_acc[l_name] += avg_mem_trains[l_name]

        #cal loss
        loss = calculate_loss(outputs, labels, network_config, layers_config, err)

        #backward
        opti.zero_grad()
        loss.backward()

        #clip grad
        clip_grad_norm_(network.module.get_parameters(), 1)
        opti.step()

        network.module.weight_clipper()

        spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
        predicted = np.argmax(spike_counts, axis=1)
        train_loss += torch.sum(loss).item()
        labels = labels.cpu().numpy()
        total += len(labels)
        correct += (predicted == labels).sum().item()


    total_accuracy = correct / total
    total_loss = train_loss / total
    if total_accuracy > glv.max_accuracy:
        glv.max_accuracy = total_accuracy
    if glv.min_loss > total_loss:
        glv.min_loss = total_loss
    return 100. * total_accuracy, total_loss, avg_mem_trains_acc

def test(network, testloader, epoch, network_config, method = None, act_eps = None):
    #test mode
    network.eval()
    
    correct = 0
    total = 0
    n_steps = network_config['n_steps']

    avg_mem_trains_acc = {l.name: 0 for l in network.module.layers if l.type in ["conv", "linear"]}

    for _, (inputs, labels) in tqdm(enumerate(testloader)):
    
        # Initialize an empty dictionary to accumulate the average spike train for each layer
        if len(inputs.shape) < 5:
            inputs = inputs.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)
        # forward pass
        labels = labels.to(device)
        inputs = inputs.to(device)

        #general adv sample
        if method == "clean": 
            adv_inputs = inputs                
        elif method =="fgm":
            adv_inputs = fast_gradient_method(network, inputs, labels, act_eps, network_config)
        elif method =="pgd":
            adv_inputs = projected_gradient_descent(network, inputs, labels, act_eps, network_config)
        elif method =="rfgm":
            adv_inputs = r_fgsm(network, inputs, labels, act_eps, act_eps/2 , network_config)
        elif method =="bim":
            adv_inputs = bim_attack(network, inputs, labels, act_eps, network_config)
        else:
            exit("wrong method")
        outputs , avg_mem_trains= network.forward(adv_inputs, False)
        
        # Accumulate the average spike train for each layer
        if(avg_mem_trains!= {}):
            for l_name in avg_mem_trains_acc.keys():
                avg_mem_trains_acc[l_name] += avg_mem_trains[l_name]


        spike_counts = torch.sum(outputs, dim=4).squeeze_(-1).squeeze_(-1).detach().cpu().numpy()
        predicted = np.argmax(spike_counts, axis=1)
        labels = labels.cpu().numpy()
        total += len(labels)
        correct += (predicted == labels).sum().item()

    # This part will gather the results from all processes.
    # First, let's gather the total counts from each process:
    total_tensor = torch.tensor([total]).to(device)
    dist.all_reduce(total_tensor)
    total = total_tensor.item()

    # Do the same for correct counts:
    correct_tensor = torch.tensor([correct]).to(device)
    dist.all_reduce(correct_tensor)
    correct = correct_tensor.item()

    # Compute the global accuracy:
    test_accuracy = correct / total

    if dist.get_rank() == 0:  # If this is the main process, update the best accuracy
        if test_accuracy > glv.best_acc:
            glv.best_acc = test_accuracy
            glv.best_epoch = epoch

    acc = 100. * test_accuracy

    return acc, avg_mem_trains_acc


def advtest(net, test_loader, params, test_only=True, ckpt=None, blackbox=False):
    atk_strength = params['ATTACK']['strength']
    attack_types = ['fgm', 'rfgm', 'pgd', 'bim']

    results = {atk: [] for atk in attack_types}

    if test_only:
        assert ckpt is not None, "Checkpoint path must be provided in test_only mode"
        checkpoint = torch.load(ckpt, map_location=device)
        net.load_state_dict(checkpoint['net'])
        print_rank0(f"Adv Testing, using {ckpt}")
    else:
        assert ckpt is None, "Checkpoint path should not be provided if not in test_only mode"
        print_rank0("Using current model!")

    # Clean test
    clean_acc, avg_mem_trains_acc = test(net, test_loader, 0, params['Network'], 'clean', 0)
    if(bool(params['Network']["save_target"])== True):
        save_target(avg_mem_trains_acc, test_loader, "test", "clean", 0)

    for atk in attack_types:
        results[atk].append(clean_acc)

    metrics = {f'{atk}_acc': clean_acc for atk in attack_types}
    wandb_log_rank0(metrics, step=1000)
    count = 1000

    # Attack test
    for ori_eps in atk_strength:
        act_eps = eval(ori_eps) / params["Network"]["std"]
        count += 1
        for atk in attack_types:
            acc, avg_mem_trains_acc = test(net, test_loader, 0, params['Network'], atk, act_eps)

            results[atk].append(acc)
            print_rank0(f"Acc test under {atk}, Ori_eps = {ori_eps}, Act_eps = {round(act_eps, 3)}, acc = {acc}")

            metrics[f'{atk}_acc'] = acc
            wandb_log_rank0(metrics, step=count)  # Assuming step increases with act_eps

            if(bool(params['Network']["save_target"])== True):
                save_target(avg_mem_trains_acc, test_loader, "test", atk, act_eps)
    # Print results
    for atk in attack_types:
        print_rank0(f"{atk} result: {results[atk]}")

    return results

def advtrain(net, train_loader, test_loader, train_sampler,
                params, mode):
    
    error = loss_f.SpikeLoss(params['Network']).to(device)
    net_parameters = net.module.get_parameters()
    learning_rate = params['Network']['lr']
    epochs = params['Network']['epochs']
    param_dict = {i: name for i, (name, _) in enumerate(net.named_parameters())}

    if mode != "clean":
        method = params['ATTACK']['ft_method']
        ori_eps = params['ATTACK']['train'][0]
        act_eps = eval(ori_eps)/params["Network"]["std"]
        print_rank0("Adv train under", method, "Ori_eps = ", ori_eps, "Act_eps = ", round(act_eps, 3))
        print_rank0("Finetune all") 
    else:
        method = "clean"
        act_eps = 0

        print_rank0("Clean Train all!")
    optimizer = torch.optim.AdamW(net_parameters, 
                                      lr=learning_rate,
                                        betas=(0.9, 0.999))    
    # 步骤2: 定义一个lr_scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, verbose= True)

    best_acc = 0
    for epoch in tqdm(range(epochs)):
        print_rank0(f"{mode} training, epoch = ", epoch)
        train_sampler.set_epoch(epoch)
        train_acc, train_loss, avg_mem_trains_acc = train(net, train_loader, optimizer, epoch, 
                                      params['Network'], params['Layers'], error, 
                                      method, act_eps)
        lr_scheduler.step()

        print_rank0("Train Accuracy: %.3f, Train Loss: %.3f " % (train_acc,train_loss))
        test_acc, _ = test(net, test_loader, epoch, params['Network'],
                        "clean", 0)
        print_rank0("Test Accuracy: %.3f"%(test_acc))
        metrics = {'train_acc':train_acc, 'train_loss':train_loss, 'test_acc':test_acc}

        wandb_log_rank0(metrics)
        
        # save ckpt
        ckpt =  params['DEFAULT']['dataset']+ params['Network']["model"] +  mode + "_" + method + "_"+ str(params['Network']["tau_v"])
        if(test_acc > best_acc):
            best_acc = test_acc
            save_ckpt(net, epoch, ckpt)
        if(bool(params['Network']["save_target"])== True):
            save_target(avg_mem_trains_acc, train_loader, "train", method, act_eps)
        if(epoch%100 ==0):
            save_ckpt(net, epoch, ckpt + str(epoch))

        dist.barrier()  # Ensure all processes have finished clean test

    advtest(net, test_loader, params, True, './checkpoint/tmp/' + ckpt + 'best.pth')
