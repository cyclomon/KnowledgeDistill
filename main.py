import numpy as np
import torch
from torch import nn, optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torch.nn import functional as F
from vgg import *
from utils import *
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from run import *
from student import *
parser = argparse.ArgumentParser(description='Knowledge Distillation')

parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--validate', action='store_true' )
parser.add_argument('--student', action='store_true')
global args
args = parser.parse_args()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='/SSD/gihyun', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='/SSD/gihyun', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers)

# val_loader = torch.utils.data.DataLoader(
#     datasets.CIFAR10(root='/SSD/gihyun', train=False, transform=transforms.Compose([
#         transforms.ToTensor(),
#         normalize,
#     ])),
#     batch_size=args.batch_size, shuffle=False,
#     num_workers=args.workers, pin_memory=True)
if args.student:
    model = Student()
    model = model.cuda()
else:
    model = vgg16_bn(pretrained=False)
    model = model.cuda()
criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if args.validate:
    validate(val_loader, model, criterion)
    

else:
    acc_arr = []
    loss_arr = []
    train_acc_arr=[]
    train_loss_arr=[]
    best_prec1=0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        prec_tr,loss_tr=train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec_val,loss_val = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec_val > best_prec1
        best_prec1 = max(prec_val, best_prec1)
        acc_arr.append(prec_val)
        loss_arr.append(loss_val)
        train_acc_arr.append(prec_tr)
        train_loss_arr.append(loss_tr)
        print('BEST_ACCURACY : {}'.format(best_prec1))
        if epoch%10 ==0:
            if args.student:
                filename=os.path.join(args.save_dir, 'checkpoint'+'student'+'_{}.pth'.format(epoch))
            else:
                filename=os.path.join(args.save_dir, 'checkpoint_{}.pth'.format(epoch))
            torch.save(model.state_dict(), filename)
            
        plt.figure()
        plt.plot(train_acc_arr)    
        plt.plot(acc_arr)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if args.student:
            plt.savefig('./accuracy_stu.png')
        else:
            plt.savefig('./accuracy_fig.png')
        # summarize history for loss
        plt.figure()
        plt.plot(train_loss_arr)
        plt.plot(loss_arr)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        if args.student:
            plt.savefig('./loss_stu.png')
        else:
            plt.savefig('./loss_fig.png')
    if args.student:    
        np.savez("./No_KD_student_result.npz", train_acc_arr=train_acc_arr, acc_arr=acc_arr, train_loss_arr=train_loss_arr, loss_arr= loss_arr )
    else:
        np.savez("./teacher_result.npz", train_acc_arr=train_acc_arr, acc_arr=acc_arr, train_loss_arr=train_loss_arr, loss_arr= loss_arr )


            

