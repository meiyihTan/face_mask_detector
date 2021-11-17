from __future__ import print_function
import os
import argparse
import time
import datetime
import math
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from data import AnnotationTransform, FaceMaskData, detection_collate, preproc, cfg
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
from models.faceboxes import FaceBoxes

parser = argparse.ArgumentParser(description='FaceBoxes Training')
parser.add_argument('--datapath', default='data/FaceMask', help='Dataset directory')
parser.add_argument('--num_classes', default=3, type=int, help='Number of classes')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--pretrained', default=False, help='use pretrained model for retraining')
parser.add_argument('--max_epoch', default=200, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='model_checkpoints/', help='Location to save checkpoint models')
args = parser.parse_args()

datapath = args.datapath
num_classes = args.num_classes
batch_size = args.batch_size
num_workers = args.num_workers
initial_lr = args.lr
momentum = args.momentum
pretrained = args.pretrained
max_epoch = args.max_epoch
weight_decay = args.weight_decay
gamma = args.gamma
save_folder = args.save_folder

img_dim = 1024 # only 1024 is supported
rgb_mean = (104, 117, 123) # bgr order

# use gpu if exists 
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
    cudnn.benchmark = True
    
if os.path.exists(save_folder)==False: 
    os.makedirs(save_folder)
    
# define model 
net = FaceBoxes('train', img_dim, num_classes, pretrained=pretrained)
net = net.to(device)

# define optimizer and loss function (criterion)
optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

# define priorbox 
# priorbox are pre-computed boxes defined at specific positions on specific feature maps, with specific aspect ratios and scales.
# they are used to match the ground-truth bouding boxes 
priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)

# training function 
def train():
    net.train()
    epoch = 0
    print('Loading Dataset...')

    dataset = FaceMaskData(datapath, 'train', preproc(img_dim, rgb_mean), AnnotationTransform())

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (200 * epoch_size, 250 * epoch_size)
    step_index = 0

    for iteration in range(max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > 200):
                torch.save(net.state_dict(), os.path.join(save_folder,'FaceBoxes_epoch_' + str(epoch) + '.pth'))
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size)

        # load train data
        images, targets = next(batch_iterator)
        images = images.to(device)
        targets = [anno.to(device) for anno in targets]

        # forward
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c
        loss.backward()
        optimizer.step()
        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        
        
        if (iteration % epoch_size)%100 == 0:         
            print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || L: {:.4f} C: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'.format(epoch, max_epoch, (iteration % epoch_size) + 1, epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))

    torch.save(net.state_dict(), os.path.join(save_folder, 'Final_FaceBoxes.pth'))


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
