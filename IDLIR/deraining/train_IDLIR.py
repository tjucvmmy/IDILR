import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from networks import *
import losses

parser = argparse.ArgumentParser(description="IDLIR_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=8, help="Training batch size")
parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[20,40,60], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=4e-4, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="/save path ", help='path to save models and log files')
parser.add_argument("--save_freq",type=int,default=1,help='save intermediate model')
parser.add_argument("--data_path",type=str, default="/date path /",help='path to training data')
parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=5, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

   
    # Build model
    model = IDLIR(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    print_network(model)

    # loss function
    criterion_1 = nn.MSELoss(size_average=True)
    criterion_2 = SSIM()
    
    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion_1.cuda()
        criterion_2.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.5)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train, = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train= input_train.cuda(), target_train.cuda()

            out_target,out_streak= model(input_train)
            
            pixel_metric = criterion_2(target_train, out_target)
            loss = criterion_1(target_train, out_target)+criterion_1(input_train-target_train, out_streak)
            loss.backward()
            optimizer.step()

            # training curve
            model.eval()
            out_target,out_streak= model(input_train)
            out_target = torch.clamp(out_target, 0., 1.)
            psnr_train = batch_PSNR(out_target, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## epoch training end

        # log the images
        model.eval()
        out_target,out_streak= model(input_train)
        out_target = torch.clamp(out_target, 0., 1.)

        im_target = utils.make_grid(out_target.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_derain = utils.make_grid(out_target.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', im_target, epoch+1)
        writer.add_image('rainy image', im_input, epoch+1)
        writer.add_image('deraining image', im_derain, epoch+1)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch+1)))


if __name__ == "__main__":
    if opt.preprocess:
        if opt.data_path.find('RainTrainH') != -1:
            print(opt.data_path.find('RainTrainH'))
            prepare_data_TrainrainH(data_path=opt.data_path, patch_size=100, stride=100)
        else:
            print('unkown datasets: please define prepare data function in DerainDataset.py')


    main()
