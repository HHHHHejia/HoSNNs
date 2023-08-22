import torch
import tools.global_v as glv


class TSSLBP(torch.autograd.Function): 
    @staticmethod
    def forward(ctx, inputs, network_config, layer_config, name, target_mem_train, theta_v ):
        device = inputs.device
        shape = inputs.shape
        n_steps = shape[4] 
        theta_m = 1/network_config['tau_m']
        tau_s = network_config['tau_s']
        theta_s = 1/tau_s

        #decide alif mode
        if(name == "output"):
            is_alif = False
        else:
            is_alif = (network_config['model'] == "ALIF")

        syns_posts = []
        mems = []
        
        with torch.no_grad():

            mem = torch.zeros(shape[0], shape[1], shape[2], shape[3]).to(device)
            syn = torch.zeros(shape[0], shape[1], shape[2], shape[3]).to(device)
            out = torch.zeros(shape[0], shape[1], shape[2], shape[3]).to(device)
            threshold = layer_config['threshold'] * torch.ones(shape[0], shape[1], shape[2], shape[3]).to(device)
            error_sum = torch.zeros(shape[0], shape[1], shape[2], shape[3]).to(device)
            error = torch.zeros(shape[0], shape[1], shape[2], shape[3]).to(device)
            if(is_alif):
                target_mem_train = target_mem_train.expand(shape[0], -1, -1, -1, -1).to(device)

            for t in range(n_steps):
                mem_update = (-theta_m) * mem + inputs[..., t]
                mem += mem_update

                #update threshold
                if(is_alif):
                    error = mem - target_mem_train[...,t]
                    threshold = threshold + theta_v * error
                    error_sum += error

                out = mem > threshold
                out = out.type(torch.float32)
                mems.append(mem)
                mem = mem * (1-out)
                syn = syn + (out - syn) * theta_s
                syns_posts.append(syn)


            mems = torch.stack(mems, dim = 4)
            syns_posts = torch.stack(syns_posts, dim = 4)
            other_tensor =  torch.tensor([layer_config['threshold'], is_alif])
        
        ctx.save_for_backward(mems, other_tensor, error_sum/n_steps)
        if(bool(network_config["save_target"]) == True):
            return syns_posts, torch.mean(syns_posts, dim=0)
        else:
            return syns_posts, None
    @staticmethod
    def backward(ctx, grad_delta, grad_outputs):
        (u, others, ei) = ctx.saved_tensors
        device = u.device
        shape = grad_delta.shape
        n_steps = shape[4]
        threshold = others[0].item()
        is_alif = bool(others[1].item())

        partial_a_tmp = glv.partial_a[..., 0, :].repeat(shape[0], shape[1], shape[2], shape[3], 1).to(device)
        grad_a = torch.empty_like(u)
        for t in range(n_steps):
            grad_a[..., t] = torch.sum(partial_a_tmp[..., 0:n_steps-t]*grad_delta[..., t:n_steps], dim = 4)

        a = 0.2
        f = torch.sigmoid(torch.clamp(-(u - threshold) / a, -8, 8))
        grad = grad_a * f * (1-f)/a
    
        if(is_alif):
            grad_v =torch.mean(-grad,dim= 0)
            grad_theta_v = 0.1*torch.mean(grad_v,dim= -1) * ei
        else:
            grad_theta_v = None

            
        return grad, None, None, None, None, grad_theta_v


