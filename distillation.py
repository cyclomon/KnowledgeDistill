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
from student import *
from run import *

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

parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--validate', action='store_true' )
parser.add_argument('--alpha',default=1,type=int)
parser.add_argument('--temp',default=3,type=int)
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

model_teacher = vgg16_bn(pretrained=False).cuda()
model_teacher.load_state_dict(torch.load('./checkpoints/checkpoint_290.pth'))

model_student = Student().cuda()

criterion = nn.CrossEntropyLoss()
criterion = criterion.cuda()

optimizer = torch.optim.SGD(model_student.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lr = args.lr * (0.5 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if args.validate:
    validate(val_loader, model_student, criterion)
    

else:
    acc_arr = []
    loss_arr = []
    train_acc_arr=[]
    train_loss_arr=[]
    best_prec1=0
    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        prec_tr,loss_tr=train_KD(train_loader, model_student,model_teacher, criterion, optimizer, epoch,args.alpha,args.temp)

        # evaluate on validation set
        prec_val,loss_val = validate(val_loader, model_student, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec_val > best_prec1
        best_prec1 = max(prec_val, best_prec1)
        acc_arr.append(prec_val)
        loss_arr.append(loss_val)
        train_acc_arr.append(prec_tr)
        train_loss_arr.append(loss_tr)
        print('BEST_ACCURACY : {}'.format(best_prec1))
        if (epoch+1)%10 ==0:
            filename=os.path.join(args.save_dir, 'checkpoint_student_KD0.01_{}.pth'.format(epoch))
            torch.save(model_student.state_dict(), filename)
            
        plt.figure()
        plt.plot(train_acc_arr)    
        plt.plot(acc_arr)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./accuracy_fig_distill0.01.png')
        # summarize history for loss
        plt.figure()
        plt.plot(train_loss_arr)
        plt.plot(loss_arr)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('./loss_fig_distill0.01.png')
    train_acc_arr = np.asarray(train_acc_arr)
    acc_arr = np.asarray(acc_arr)
    train_loss_arr = np.asarray(train_loss_arr)
    loss_arr = np.asarray(loss_arr)
    
    np.savez("./KD_student_result.npz", train_acc_arr=train_acc_arr, acc_arr=acc_arr, train_loss_arr=train_loss_arr, loss_arr= loss_arr )



            

