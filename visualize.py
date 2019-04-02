import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import attacks
import numpy as np
import example
from models.vgg import VGG
from models.lenet import LeNet

use_cuda = torch.cuda.is_available()

attacker = attacks.FGSM()
# attacker.load('saved/VGG16_attacker_0.005.pth')

model = VGG('VGG16')
model.cuda()
model = torch.nn.DataParallel(
    model, device_ids=range(torch.cuda.device_count()))
model.load_state_dict(torch.load('saved/VGG16.pth'))

criterion = nn.CrossEntropyLoss()
trainloader, testloader = example.load_cifar()

for inputs, labels in testloader:
    inputs = Variable((inputs.cuda() if use_cuda else inputs),
                      requires_grad=True)
    labels = Variable((labels.cuda() if use_cuda else labels),
                      requires_grad=False)
    adv_inputs, i, j = attacker.attack(inputs, labels, model)
    vutils.save_image(inputs.data, 'images/VGG_unperturbed.png')
    vutils.save_image(adv_inputs, 'images/VGG_FGSM.png')
    break
