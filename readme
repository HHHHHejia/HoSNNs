# HoSNNs
This is code for http://arxiv.org/abs/2308.10373, HoSNN: Adversarially-Robust Homeostatic Spiking Neural Networks with
Adaptive Firing Thresholds

We uploaded two branches of codes, namely HoSNN(mnist, cifar10) and HoSNN-cifar100(for CIFAR100). They are almost exactly the same, the only difference is that cifar100 directly uses the classification result of the last layer and uses softmax to calculate the loss ( For better learning effect), mnist and cifar10 use the last layer of lif neurons to output psc and the kernel loss of the target psc to calculate the loss. For both code, what you need to do is almost the same.

Besides, For attack on MNIST, we remove the clamp to [0,1] to deliver stronger attack, in attack.py
For cifar10 and cifar100, won't make a difference.

Some pretrained model:
CIFAR10, HoSNN with fgsm 2/255 training: 
https://drive.google.com/file/d/1TIWMtMWKai-1samX_87L9YqHj-fUKPgf/view

CIFAR100, HoSNN with fgsm 4/255 training: 
https://drive.google.com/file/d/17zcFieB_BCrTaOQzLj13gLsRUwsCkBiG/view

```plaintext
----------------------------------------------------
1.first install the dependencies:

pip install -r requirements.txt

--------------------------------------------------

2.Please use the following commands to run the code in a distributed mode, 
and you can freely choose the number of GPUs you use. 
Eg, you're using your 0,1 gpu, so "CUDA_VISIBLE_DEVICES=0,1", and "--nproc_per_node=2":

For MNIST, -d should be 0: 
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29506  --nproc_per_node=2 main.py -d 0

For CIFAR10, -d should be 1: 
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29506  --nproc_per_node=2 main.py -d 1

For CIFAR100, -d should be 2: 
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:29506  --nproc_per_node=2 main.py -d 2

------------------------------------------------------


3.All the running options are placed in the yaml file in the networks folder, and I will choose an example below. Eg for MNIST:<br>

DEFAULT:
  dataset: mnist 		#(you don't need to change)  
  ckpt: 			#(os dirpath. path for model checkpoint)
  cleantrain: True		#(True or False. If True, train a net from scratch)
  advtrain: False               #(True or False. If True, adversarial train a net, need the ckpt model)
  advtest: False		#(True or False. If True, adversarial test the ckpt model)
	
Network:
  epochs: 201
  model: ALIF			#(LIF or ALIF; ALIF is the TA-LIF mode in the paper)
  tau_v: 1.5			#(a positive real number from 1 to inf as tauv in the paper)
  ckpt_v: ./spikedata/mnist/mnist.pt #(os dirpath. path for the NDS, i have prepared a file for you)
  batch_size: 64
  lr: 0.0005
  is_bn: False			#(True or False, whether to use BN layer)
  save_target: False		#(True or False, whether to save your NDS)
  n_steps: 5			#simulation time
  data_path: ./datasets/mnist	
  mean: 0.1307			#mean of the dataset
  std: 0.3081			#std of the dataset
  dataset: MNIST
  loss: "kernel"		#loss, "kernel" for MNIST,CIFAR10 and "softmax" for CIFAR100, don't need to change
  n_class: 10			
  tau_m: 5			#membrane constant
  tau_s: 3			#psc constant

ATTACK:
  strength: [1/10, 2/10, 3/10, 4/10] #during adv testing,  the strength eps to use
  ft_method: fgm		#during adv training,  the adv method to use
  train: [1/10]			#during adv training,  the eps to use

Layers:
  ...(omitted)
```
