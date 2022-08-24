import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import sys
from torch.utils.data import Dataset,DataLoader
sys.path.append("../../")
from model import TCN
import numpy as np
import random
from utils import BreatheDataset
from os import walk
from os import path
import matplotlib.pyplot as plt
from torchtext.data.functional import to_map_style_dataset

parser = argparse.ArgumentParser(description='Sequence Modeling - lung size prediction')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout applied to layers (default: 0.1)')
parser.add_argument('--clip', type=float, default=0,
                    help='gradient clip, -1 means no clip (default: 0)')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit (default: 100)')
parser.add_argument('--ksize', type=int, default=5,
                    help='kernel size (default: 5)')
parser.add_argument('--levels', type=int, default=5,
                    help='# of levels (default: 5)') #log base 2((375-1)/(9-1)) = 5.54
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='report interval (default: 20')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=32,
                    help='number of hidden units per layer (default: 32')
parser.add_argument('--data', type=str, default='Aligned_Array',
                    help='the dataset to run (default: Aligned_Array)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')

def evaluate(dataset, name='Eval'):
    model.eval()
    total_batch_loss = 0.0 #to save the sum of all batch loss
    count = 0 # count how many batch
    with torch.no_grad():
        for i,(one_batch_data) in enumerate(dataset):
            x, y = one_batch_data
            x, y = Variable(x), Variable(y)
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            output = model(x)
            loss = criterion(output,y)
            loss += loss.item()
            total_batch_loss += loss.item()
            count += 1
        # print(total_batch_loss) #0
        # print(count) #0.0
        eval_loss = total_batch_loss / count #epoch loss
        print(name + " loss: {:.5f}".format(eval_loss))
        return eval_loss

def train(train_dataset, ep):
    model.train()
    total_batch_loss = 0.0
    count = 0
    for i,(one_batch_train_data) in enumerate(train_dataset):
        x, y = one_batch_train_data
        x, y = Variable(x), Variable(y)
        if args.cuda:
            x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        total_batch_loss += loss.item()
        count += 1

        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        loss.backward()
        optimizer.step()
        if i > 0 and i % args.log_interval == 0:
            cur_loss = total_batch_loss / count
            print("Epoch {:2d} | lr {:.5f} | loss {:.5f}".format(ep, lr, cur_loss))
            total_batch_loss = 0.0
            count = 0

def draw_result(iter: int, lst_loss: list, title: str):
    lst_iter = range(iter)
    plt.plot(lst_iter, lst_loss, '-b', label='loss')
    plt.xlabel("n iteration")
    plt.legend(loc='upper left')
    plt.title(title)
    # save image
    plt.savefig(title+".png")  # should before show method
    # show
    plt.show()

if __name__ == "__main__":
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    print(args)
    input_size = 1
    output_size = 1
    train_files = []
    test_files = []
    validation_files = []
    for dirPath, dirNames, fileNames in walk("processed_data\\train"):
        for f in fileNames:
            train_files.append((path.join(dirPath, f)))
    for dirPath, dirNames, fileNames in walk("processed_data\\test"):
        for f in fileNames:
            test_files.append((path.join(dirPath, f)))
    for dirPath, dirNames, fileNames in walk("processed_data\\val"):
        for f in fileNames:
            validation_files.append((path.join(dirPath, f)))

    train_dataset = to_map_style_dataset(BreatheDataset(train_files, 2 ,16 ,random_seed=123))
    test_dataset = to_map_style_dataset(BreatheDataset(test_files, 2 ,16 ,random_seed=123))
    val_dataset = to_map_style_dataset(BreatheDataset(validation_files, 2 ,16 ,random_seed=123))
    train_data_loader = DataLoader(dataset=train_dataset, num_workers=0, batch_size=None, shuffle=False)
    val_data_loader = DataLoader(dataset=val_dataset, num_workers=0, batch_size=None,  shuffle=False )
    test_data_loader = DataLoader(dataset=test_dataset, num_workers=0, batch_size=None, shuffle=False )

    n_channels = [args.nhid] * args.levels
    kernel_size = args.ksize
    dropout = args.dropout
    model = TCN(input_size, output_size, n_channels, kernel_size, dropout=args.dropout).double()

    if args.cuda:
        model.cuda()

    criterion = nn.MSELoss(reduction='mean') ##
    lr = args.lr
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)

    best_vloss = 1e8
    vloss_list = []
    tloss_list = []
    model_name = "breath_{0}.pt".format(args.data)
    for ep in range(1, args.epochs+1):
        train(train_data_loader,ep)
        vloss = evaluate(val_data_loader, name='Validation')
        tloss = evaluate(test_data_loader, name='Test')
        if vloss < best_vloss:
            with open(model_name, "wb") as f:
                torch.save(model.state_dict(), f)
                print("Saved model!\n")
            best_vloss = vloss
        if ep > 100 and vloss > max(vloss_list[-10:]):
            lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        vloss_list.append(vloss)
        tloss_list.append(tloss)


    print('-' * 89)
    draw_result(len(vloss_list), vloss_list, 'val_loss')
    draw_result(len(tloss_list), tloss_list, 'test_loss')
    model.load_state_dict(torch.load(model_name))
    tloss = evaluate(test_data_loader, name='test')

