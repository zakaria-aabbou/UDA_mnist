############################ Imports ###################################
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
import argparse
import torchvision
from datetime import datetime
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from ema import EMA
from model import Wide_ResNet, VanillaNet
from randAugment import RandAugment, myRandAugment
from data import CIFAR10Sup, CIFAR10Unsup, CIFAR10Val
from data_MNIST import MNIST_Sup , MNIST_Unsup , MNIST_Val

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

########################## Global setting ##############################
'''
Global setting is initialized here:
    - Hyper-prameters (dumped in output directory)
    - Ouput directory
    - Tensorboard writer
'''

MOD = 'semisup'                 # Supervised (sup) or semi-supervised training (semisup)
SUP_NUM = 100                   # default=4000, Number of samples in supervised training set (out of 50K)
VAL_NUM = 20 #1000                  # default=1000, Number of samples in validation set (out of 50K)
RAND_SEED = 89                  # default=89 Random seed for dataset shuffle
SUP_AUG   = ['crop', 'hflip']   # Valid values: crop, hflip, cutout, randaug
UNSUP_AUG = ['randaug']         # Valid values: crop, hflip, cutout, randaug
BSZ_SUP = 20 #64                    # Batch size for supervised training
BSZ_UNSUP = 20 #448                 # Batch size for unsupervised training
SOFTMAX_TEMP = 0.4              # Softmax temperature for target distribution (unsup)
CONF_THRESH = 0.5               # Confidence threshold for target distribution (unsup)
UNSUP_LOSS_W = 1.0              # Unsupervised loss weight
MAX_ITER = 20 #500000               # Total training iterations
VIS_IDX = 20                    # Output visualization index
EVAL_IDX = 20 #1000                 # Validation index
OUT_DIR = './output/'           # Output directory

OUT_DIR = '{}{}/'.format(OUT_DIR ,  datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
MODEL_PATH = '{}best_model.pth'.format(OUT_DIR)

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

# with open('{}args.txt'.format(OUT_DIR), 'w') as f:
#     json.dump(__dict__, f, indent=2)

if MOD == 'semisup':
    print("mode is semisup !!\n")
    #assert SUP_NUM == 4000, "Remove assertion if you wish to have semi sup training with sup set != 4K"

if MOD == 'sup':
    print("mode is sup !!\n")
    # assert SUP_NUM == 49000, "Remove assertion if you wish to have sup training with sup set != 49K"

writer = SummaryWriter(OUT_DIR)

######################## Data initialization ###########################
'''
Input data is initialized here, along with the train (sup & unsup), valid and test dataloaders:
    - transform_train_sup contains the list of transformations (input params) to be applied to supervised and unsupervised samples.
    - transform_train_unsup contains the list of transformations (input params) to be applied to unsupervised samples (noise injection).
    - transform_test contains the list of transformations (tensor & norm) to be applied to valid and test samples.
'''

SUP_AUG   += ["tensor", "normalize"]
UNSUP_AUG += ["tensor", "normalize"]

transforms_aug = {"crop": transforms.RandomCrop(28, padding=4, padding_mode="reflect"),
                  "hflip": transforms.RandomHorizontalFlip(),
                  "cutout": transforms.RandomErasing(value='random'),
                  "randaug": RandAugment(2, 15),
                  "tensor": transforms.ToTensor(),
                  "normalize": transforms.Normalize(
                      #(0.49138702, 0.48217663, 0.44645257),
                      #(0.24706201, 0.24354138, 0.2616881)
                      [0.5],[0.5]
                      )
                }

transform_train_sup = transforms.Compose(
    [transforms_aug[val] for val in SUP_AUG])
transform_train_unsup = transforms.Compose(
    [transforms_aug[val] for val in UNSUP_AUG])
transform_test = transforms.Compose(
    [transforms_aug[val] for val in ["tensor", "normalize"]])

# trainset_sup = CIFAR10Sup(root='./data', train=True, download=True, transform=[
#                           transform_train_sup], sup_num=SUP_NUM, random_seed=RAND_SEED)
# trainset_unsup = CIFAR10Unsup(root='./data', train=True, download=True, transform=[
#                               transform_train_sup, transform_train_unsup], sup_num=SUP_NUM, random_seed=RAND_SEED)
# validset = CIFAR10Val(root='./data', train=True, download=True, transform=[
#                       transform_test], val_num=VAL_NUM, random_seed=RAND_SEED)
# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)


trainset_sup = MNIST_Sup(root='./data', train=True, download=True, transform=[
                          transform_train_sup], sup_num=SUP_NUM, random_seed=RAND_SEED)
trainset_unsup = MNIST_Unsup(root='./data', train=True, download=True, transform=[
                              transform_train_sup, transform_train_unsup], sup_num=SUP_NUM, random_seed=RAND_SEED)
validset = MNIST_Val(root='./data', train=True, download=True, transform=[
                      transform_test], val_num=VAL_NUM, random_seed=RAND_SEED)
testset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform_test)




trainloader_sup = torch.utils.data.DataLoader(
    trainset_sup, batch_size=BSZ_SUP, num_workers=2, drop_last=True)
trainloader_unsup = torch.utils.data.DataLoader(
    trainset_unsup, batch_size=BSZ_UNSUP, num_workers=2, drop_last=True)
validloader = torch.utils.data.DataLoader(
    validset, batch_size=4, shuffle=False, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=4, shuffle=False, num_workers=2)

classes = ('0', '1', '2', '3', '4',
           '5', '6', '7', '8', '9')


######################### Visualize data ###############################
'''
Some input samples are visualized here:
    - saved in output directory
    - plotted on tensorboard
'''

from IPython.display import display
from PIL import Image

def unnormalize(img):
    mean = torch.Tensor([0.5]).unsqueeze(-1)
    std = torch.Tensor([0.5]).unsqueeze(-1)
    img = (img.view(2, -1) * std + mean).view(img.shape)
    img = img.clamp(0, 1)
    return img


def save_grid(img):
    npimg = img.numpy()
    path = '{}in_data.jpg'.format(OUT_DIR)
    plt.imsave(path, np.transpose(npimg, (1, 2, 0)))
    display(Image.open(path))



dataiter = iter(trainloader_sup)
images, labels = dataiter.next()
images_grid = torchvision.utils.make_grid(images)
images_grid = unnormalize(images_grid)
save_grid(images_grid)
writer.add_image('input_images', images_grid, 0)


############################# Model ####################################
'''
Classification model is initialized here, along with exponential
moving average (EMA) module:
    - model is pushed to gpu if its available.
'''

net = Wide_ResNet(28, 2, 0.3, 10)  # VanillaNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)
ema = EMA(net, decay=0.9999)


############################## Utils ###################################
'''
Training utils are initialized here, including:
    - CrossEntropyLoss - supervised loss.
    - KLDivLoss - unsupervised consistency loss
    - SGD optimizer
    - CosineAnnealingLR scheduler
    - Evaluation function
'''

criterion_sup = torch.nn.CrossEntropyLoss()
criterion_unsup = torch.nn.KLDivLoss(reduction='none')
optimizer = torch.optim.SGD(
    net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, MAX_ITER)


def eval_model(model, valloader, write, writer_id):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on validation set: %.2f %%' %
          (100.0 * correct / total))
    write.add_scalar('validation/Accuracy', 100.0 * correct / total, writer_id)
    model.train()
    return correct


############################ Training ##################################
'''
Training loop containing:
    - data loading
    - optimizer initialization
    - fixed model parameters to generate unsup target logits
    - prediction sharpening of unsup target logits
    - confidence threshold of unsup logits
    - supervised cross entropy loss
    - unsupervised consistency loss
    - exponential moving average of model parameters
    - printing/plotting of the training stats
    - model evaluation every EVAL_IDX iterations
'''

running_loss = [0.0, 0.0, 0.0]
best_val = 0

trainloader_sup_iter = iter(trainloader_sup)
if MOD== 'semisup':
    trainloader_unsup_iter = iter(trainloader_unsup)

for train_idx in range(MAX_ITER):
    # data loading
    img_sup, labels_sup = trainloader_sup_iter.next()
    img_sup, labels_sup = img_sup.to(device), labels_sup.to(device)

    if MOD== 'semisup':
        img_unsup, img_unsup_aug = trainloader_unsup_iter.next()
        img_unsup, img_unsup_aug = img_unsup.to(
            device), img_unsup_aug.to(device)
        img_in = torch.cat([img_sup, img_unsup_aug])
    else:
        img_in = img_sup

    # optimizer initilization
    optimizer.zero_grad()

    if MOD== 'semisup':
        # fixed parameters of the model to stop gradient back propagation
        with torch.no_grad():
            logits_unsup = net(img_unsup)
            # prediction sharpening
            logits_unsup = logits_unsup / SOFTMAX_TEMP
            # confidence threshold (mask)
            conf_mask = F.softmax(logits_unsup, dim=1).max(dim=1)[
                0] > CONF_THRESH

    img_out = net(img_in)
    # supervised loss
    logits_sup = img_out[:BSZ_SUP]
    loss_sup = criterion_sup(logits_sup, labels_sup)

    if MOD== 'semisup':
        if conf_mask.sum() > 0:
            # Unsupervised consistency loss
            logits_unsup_aug = img_out[BSZ_SUP:]
            loss_unsup = criterion_unsup(F.log_softmax(
                logits_unsup_aug, dim=1), F.softmax(logits_unsup, dim=1))
            loss_unsup = loss_unsup[conf_mask]
            loss_unsup = loss_unsup.sum(dim=1).mean()
        else:
            loss_unsup = 0
        loss = loss_sup + (loss_unsup * UNSUP_LOSS_W)
    else:
        loss = loss_sup

    # train optimization
    loss.backward()
    optimizer.step()
    scheduler.step()

    # exponential moving average
    ema(net, train_idx // (BSZ_SUP+BSZ_UNSUP))

    # print/plot stats
    running_loss[0] += loss.item()
    running_loss[1] += loss_sup.item()
    if MOD== 'semisup':
        loss_unsup = loss_unsup.item() if type(
            loss_unsup) == torch.Tensor else loss_unsup
        running_loss[2] += loss_unsup

    writer.add_scalar(
        'learning_rate', optimizer.param_groups[0]['lr'], train_idx)
    if train_idx % VIS_IDX == VIS_IDX-1:
        writer.add_scalar('training/total_loss', loss.item(), train_idx)
        writer.add_scalar('training/sup_loss', loss_sup.item(), train_idx)
        if MOD== 'semisup':
            writer.add_scalar('training/unsup_loss', loss_unsup, train_idx)
            print('[%d] loss: %.3f loss_sup: %.3f loss_unsup: %.3f' % (
                train_idx, running_loss[0] / 100, running_loss[1] / 100, running_loss[2] / 100))
        else:
            print('[%d] loss: %.3f loss_sup: %.3f' %
                  (train_idx, running_loss[0] / 100, running_loss[1] / 100))
        running_loss = [0.0, 0.0, 0.0]

    # eval model
    if train_idx % EVAL_IDX == EVAL_IDX-1:
        ema.assign(net)
        curr_val = eval_model(net, validloader, writer, train_idx)
        ema.resume(net)
        # save model
        if curr_val > best_val:
            torch.save(net.state_dict(), MODEL_PATH)

    # impose infinite loop
    if train_idx % trainloader_sup_iter.__len__() == trainloader_sup_iter.__len__()-1:
        trainloader_sup_iter = iter(trainloader_sup)
        if MOD== 'semisup':
            trainloader_unsup_iter = iter(trainloader_unsup)

print('Finished Training')

######################### Model loading ################################
'''
Model loading:
    - Not necessary but kept as it was in the starting code.
'''

net = Wide_ResNet(28, 2, 0.3, 10)
net.load_state_dict(torch.load(MODEL_PATH))

############################# Testing ##################################
'''
Testing loop:
    - kept as it was in the starting code.
'''

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %.2f %%' %
      (100.0 * correct / total))
writer.add_scalar('testing/Accuracy', 100.0 * correct / total, 0)

############################ Class stats ###############################
'''
Class level results:
    - kept as it was in the starting code.
'''

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %.2f %%' %
          (classes[i], 100.0 * class_correct[i] / class_total[i]))
    writer.add_scalar(
        'testing/Accuracy/{}'.format(classes[i]), 100.0 * class_correct[i] / class_total[i], 0)



