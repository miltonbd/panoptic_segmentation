import sys, os
gpu=1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm

from ptsemseg.models import get_model
from ptsemseg.loader import get_loader, get_data_path
from ptsemseg.metrics import runningScore
from ptsemseg.loss import *
from ptsemseg.augmentations import *
from tensorboardX import SummaryWriter
log_dir='logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir)
best_iou=0

def train(args):
    best_saved_model_name = "{}_{}_best_model.pkl".format(args.arch, args.dataset)
    # Setup Augmentations
    data_aug= Compose([RandomRotate(10),                                        
                       RandomHorizontallyFlip()])

    # Setup Dataloader
    data_loader = get_loader(args.dataset)
    data_path = get_data_path(args.dataset)
    t_loader = data_loader(data_path, is_transform=True, img_size=(args.img_rows, args.img_cols), augmentations=data_aug, img_norm=args.img_norm)
    v_loader = data_loader(data_path, is_transform=True, split='val', img_size=(args.img_rows, args.img_cols), img_norm=args.img_norm)

    n_classes = t_loader.n_classes
    trainloader = data.DataLoader(t_loader, batch_size=args.batch_size, num_workers=8, shuffle=True)
    valloader = data.DataLoader(v_loader, batch_size=args.batch_size, num_workers=8)

    # Setup Metrics
    running_metrics = runningScore(n_classes)
        
    # Setup visdom for visualization
    if args.visdom:
        vis = visdom.Visdom()

        loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='minibatches',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))

    # Setup Model
    model = get_model(args.arch, n_classes)
    
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model.cuda()
    
    # Check if model has custom optimizer / loss
    if hasattr(model.module, 'optimizer'):
        optimizer = model.module.optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.l_rate, momentum=0.99, weight_decay=5e-4)

    if hasattr(model.module, 'loss'):
        print('Using custom loss')
        loss_fn = model.module.loss
    else:
        loss_fn = cross_entropy2d

    start_epoch=0
    try:
        if os.path.isfile(best_saved_model_name):
            print("Loading model and optimizer from checkpoint '{}'".format(best_saved_model_name))
            checkpoint = torch.load(best_saved_model_name)
            model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            start_epoch=checkpoint['epoch']
            print("Checkpoint Loaded Successfully: '{}' (epoch {})"
                  .format(best_saved_model_name, start_epoch))
    except Exception as e:
            print("No checkpoint found at '{}'".format(e))

    best_iou = -100.0 
    for epoch in range(start_epoch, args.n_epoch):
        model.train()
        for i, (images, labels) in enumerate(trainloader):
            # if i>10:
            #     break
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
            step = epoch * len(trainloader) + i
            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(input=outputs, target=labels)

            loss.backward()
            optimizer.step()

            batch_loss = loss.data[0]
            if args.visdom:
                vis.line(
                    X=torch.ones((1, 1)).cpu() * i,
                    Y=torch.Tensor([batch_loss]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')
            writer.add_scalar("train loss",batch_loss,step)
            if (i+1) % 20 == 0:
                print("Epoch [%d/%d] Loss: %.4f" % (epoch + 1, args.n_epoch, batch_loss))

        model.eval()
        for i_val, (images_val, labels_val) in tqdm(enumerate(valloader)):
            # if i > 10:
            #     break
            images_val = Variable(images_val.cuda(), volatile=True)
            labels_val = Variable(labels_val.cuda(), volatile=True)

            outputs = model(images_val)
            pred = outputs.data.max(1)[1].cpu().numpy()
            gt = labels_val.data.cpu().numpy()
            running_metrics.update(gt, pred)

        (acc,acc_cls,fwavacc,mean_iu), class_iou = running_metrics.get_scores()
        # {'Overall Acc: \t': acc,
        #  'Mean Acc : \t': acc_cls,
        #  'FreqW Acc : \t': fwavacc,
         # 'Mean IoU : \t': mean_iu}
        writer.add_scalar('Accuracy',acc,epoch)
        writer.add_scalar('Accuracy class',acc_cls,epoch)
        writer.add_scalar('fwavacc',fwavacc,epoch)
        writer.add_scalar('Mean IOU',mean_iu,epoch)

        running_metrics.reset()

        if mean_iu >= best_iou:
            print("\n==> Saving best iou:{} over previous:{}".format(mean_iu,best_iou))
            best_iou = mean_iu
            state = {'epoch': epoch+1,
                     'model_state': model.state_dict(),
                     'optimizer_state' : optimizer.state_dict(),
                     'iou':mean_iu}
            torch.save(state, best_saved_model_name)
F
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--arch', nargs='?', type=str, default='fcn8s',
                        help='Architecture to use [\'fcn8s, unet, segnet, pspnet etc\']')
    parser.add_argument('--dataset', nargs='?', type=str, default='pascal', 
                        help='Dataset to use [\'pascal, camvid, ade20k etc\']')
    parser.add_argument('--img_rows', nargs='?', type=int, default=256, 
                        help='Height of the input image')
    parser.add_argument('--img_cols', nargs='?', type=int, default=256, 
                        help='Width of the input image')

    parser.add_argument('--img_norm', dest='img_norm', action='store_true', 
                        help='Enable input image scales normalization [0, 1] | True by default')
    parser.add_argument('--no-img_norm', dest='img_norm', action='store_false', 
                        help='Disable input image scales normalization [0, 1] | True by default')
    parser.set_defaults(img_norm=True)

    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, 
                        help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--feature_scale', nargs='?', type=int, default=1, 
                        help='Divider for # of features to use')
    parser.add_argument('--resume', nargs='?', type=str, default=None,    
                        help='Path to previous saved model to restart from')

    parser.add_argument('--visdom', dest='visdom', action='store_true', 
                        help='Enable visualization(s) on visdom | False by default')
    parser.add_argument('--no-visdom', dest='visdom', action='store_false', 
                        help='Disable visualization(s) on visdom | False by default')
    parser.set_defaults(visdom=False)

    args = parser.parse_args()
    train(args)
