# -*- coding: utf-8 -*- 
# @Time : 2020/5/13 14:12 
# @Author : lxd 
# @File : train.py
import torch
from torch.optim import lr_scheduler
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter
# from visdom import Visdom


def train(model, train_loader, test_loader, step=1, epochs=500, lr=0.01, loss_fn=None, optim_fn=None, use_cuda=False):
    if use_cuda:
        device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')
    print('device:', device)


    model = model.to(device)
    if not loss_fn:
        loss_fn = torch.nn.MultiLabelSoftMarginLoss()
    if not optim_fn:
        optim_fn = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = lr_scheduler.StepLR(optim_fn, step_size=7, gamma=0.1)
    best_loss = 1
    best_acc = 0.85
    writer = SummaryWriter('../logs/tensorboard')
    # viz = Visdom()
    # assert viz.check_connection()
    # viz.line([[0., 0., 0., 0.]], [0], win='train&test',
    #          opts=dict(title='loss&acc', legend=['train_loss', 'train_acc', 'test_loss', 'test_acc']))
    for epoch in range(epochs):
        loss_count = 0
        acc = 0
        count = 0
        start = time.time()
        scheduler.step()
        for i, (x, y) in enumerate(train_loader):
            # x = x.to(device)
            out = model(x.to(device))
            # print(out.size())
            # print(y.size())
            loss = loss_fn(out, y.to(device))
            optim_fn.zero_grad()
            loss.backward()
            optim_fn.step()
            loss = loss.cpu().item()
            loss_count += loss * x.size(0)
            out = out.cpu().reshape(out.size(0), 4, -1)
            y = y.reshape(y.size(0), 4, -1)
            acc += (torch.argmax(out, 2) == torch.argmax(y, 2)).all(1).sum().item()
            count += x.size(0)
            if i % step == 0:
                print(f'epoch: {epoch}, step: {i}, loss: {loss_count / count}')
        train_loss = loss_count / count
        train_acc = acc / count
        if epoch % 1 == 0:
            loss, acc = 0, 0
            count = 1e-6
            for i, (x, y) in enumerate(test_loader):
                with torch.no_grad():
                    out = model(x.to(device))
                    loss += (loss_fn(out, y.to(device)).mean()).cpu().item() * x.size(0)
                    count += x.size(0)
                    out=out.cpu()
                    out = out.reshape(out.size(0), 4, -1)
                    y = y.reshape(y.size(0), 4, -1)
                    acc += (torch.argmax(out,2)==torch.argmax(y,2)).all(1).sum().item()
            print(f'epoch: {epoch}, loss: {loss/count}, acc: {acc/count}, time: {time.time()-start}')
            if best_loss > loss / count:
                best_loss = loss / count
                print(f'best_loss: {best_loss}, *loss')
            if best_acc < acc / count:
                best_acc = acc / count
                print(f'best_acc: {best_acc} *acc')
                torch.save(model, f'../models/Complete/model{best_acc:.2f}.pkl')
                torch.save(model.state_dict(), f'../models/Parameter/model{best_acc:.2f}.pkl')

            test_loss = loss / count
            test_acc = acc / count
            writer.add_scalar('loss', test_loss)
        # # 更新串口图像
        # viz.line([[train_loss, test_loss]],
        #          [epoch], win='loss', update='append',
        #          opts=dict(title='loss',
        #                    legend=['train_loss', 'test_loss']))
        # viz.line([[train_acc, test_acc]],
        #          [epoch], win='acc', update='append',
        #          opts=dict(title='acc',
        #                    legend=['train_acc', 'test_acc']))

