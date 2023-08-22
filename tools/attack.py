import torch
import torch.nn as nn
from functions.loss_f import psp


criterion = nn.CrossEntropyLoss()
#for mnist, we remove the clamp to dilever stronger adversarial attack
is_clamp = False
#for cifar10 and cifar100, would not be significantly different

def normalize_parameters(mean, std):
    zero_normed = -mean/std 
    max_normed = (1-mean)/std 
    if is_clamp:
        return zero_normed, max_normed
    else:
        return -float("inf"), float("inf")


# Fast Gradient Method (FGM) 
def fast_gradient_method(model_fn, images, targets, eps = None, network_config=None):

    mean = network_config["mean"]
    std = network_config["std"]
    zero_normed, max_normed = normalize_parameters(mean, std)
    images = images.clone().detach().requires_grad_(True)
    targets = targets.clone().detach()

    # cal snn loss
    targets = psp(targets, network_config).detach()
    outputs, _ = model_fn(images,False)
    loss = criterion(outputs, targets)

    # update image
    grad = torch.autograd.grad(loss, images, retain_graph=False, create_graph=False)[0]
    adv_images = torch.clamp(images + eps * grad.sign(), zero_normed, max_normed).detach()

    return adv_images


# Projected Gradient Descent (PGD) 
def projected_gradient_descent(model_fn, images, targets, eps = None, network_config=None, alpha= None, steps= 7):
    mean = network_config["mean"]
    std = network_config["std"]
    zero_normed, max_normed = normalize_parameters(mean, std)

    alpha = eps/3
    images = images.clone().detach().requires_grad_(True)

    adv_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        # call FGM
        adv_images = fast_gradient_method(model_fn, adv_images, targets, alpha, network_config)
        # Clip to ensure within eps
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp((images + delta), zero_normed, max_normed).detach()
    return adv_images

# Gaussian Noise Attack
def gaussian_noise_attack(images, eps, network_config):
    mean = network_config["mean"]
    std = network_config["std"]
    zero_normed, max_normed = normalize_parameters(mean, std)
    adv_images = torch.clamp(images + eps * torch.randn_like(images), zero_normed, max_normed).detach()
    return adv_images

#rfgsm
def r_fgsm(model_fn, images, targets, eps, alpha, network_config=None):
    mean = network_config["mean"]
    std = network_config["std"]
    zero_normed, max_normed = normalize_parameters(mean, std)

    # Perturb the images with random noise
    images_new = images + alpha * torch.randn_like(images).sign()
    images_new.requires_grad = True
    
    # Get the model's prediction
    outputs, _ = model_fn(images_new, False)
    
    # Calculate the loss
    targets = psp(targets, network_config).detach()  # Apply the PSP function as seen in your previous code
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Apply the FGSM perturbation
    attack_images = torch.clamp(images_new + (eps - alpha) * images_new.grad.sign(), zero_normed, max_normed).detach()
    
    return attack_images

#bim attack

# Basic Iterative Method (BIM) Attack
def bim_attack(model_fn, images, targets, eps=None, network_config=None, steps=7):
    # Compute alpha based on the relation alpha * steps = eps
    mean = network_config["mean"]
    std = network_config["std"]
    zero_normed, max_normed = normalize_parameters(mean, std)

    alpha = eps / steps
    images = images.clone().detach().requires_grad_(True)
    adv_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        
        # Step 1: Apply the FGM with the computed alpha
        adv_images = fast_gradient_method(model_fn, adv_images, targets, alpha, network_config)
        
        # Step 2: Project the perturbed image to ensure it's within the epsilon boundary
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images =torch.clamp((images + delta), zero_normed, max_normed).detach()

    return adv_images
