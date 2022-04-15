import warnings
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

from DataSet import BATCH_SIZE, train_loader, valid_loader
from AcousNet import *

writer = SummaryWriter('runs/MyNet_V1_0924_500epoch')

# Manually set lr attenuation interval
def adjust_learning_rate(optimizer, epoch):

    if epoch <  EPOCH/5 * 1:
            lr = 0.005
            adjust_learning_rate
    elif epoch < EPOCH/5 * 2:
        lr = 0.001
    elif epoch < EPOCH/5 * 3:
        lr = 0.0005
    elif epoch < EPOCH/5 * 4:
        lr = 0.0001
    else:
        lr = 0.00005

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Cosine based loss function
class Custom_LossFunction(nn.Module):
    def __init__(self):
        super(Custom_LossFunction, self).__init__()

    def forward(self, Est, Truth):

        # Step1 & 2: Calculate residual between Truth and Est and convert them to [0,2*pi]
        # TwoPi = np.ones((BATCH_SIZE, 2500)) * 2 * np.pi
        # TwoPi = torch.from_numpy(TwoPi).to(device)
        Residual_abs = torch.abs(Truth - Est)
        Residual = 2 * np.pi * Residual_abs   # [0, 2pi]

        # print('minimum Residual between Truth and Est: {:4f}'.format(Residual.min()))
        # print('maximum Residual between Truth and Est: {:4f}'.format(Residual.max()))

        # Step 3: Calculate the cosin value of the residual
        Residual_cos = torch.cos(Residual)  # [-1, 1]
        # print('minimum Cosin Residual between Truth and Est: {:4f}'.format(Residual_cos.min()))
        # print('maximum Cosin Residual between Truth and Est: {:4f}'.format(Residual_cos.max()))

        # Step 4: Calculate I - Residual_cos
        Residual_cos_size = list(Residual_cos.size())
        I = np.ones((Residual_cos_size[0], 2500))
        I = torch.from_numpy(I).to(device)
        Distance = I - Residual_cos   # [0, 2]
        Distance_min = Distance.min()
        Distance_max = Distance.max()
        # print('minimum Distance between Truth and Est: {:4f}'.format(Distance_min))
        # print('maximum Distance between Truth and Est: {:4f}'.format(Distance_max))

        # Step 5: Calculate loss value
        Loss = torch.mean(Distance)

        return Loss, Distance_min, Distance_max


if __name__ == '__main__':

    # n_train = len(both_dataset) * 0.8
    # n_test = len(both_dataset) * 0.2
    # print("using {} Y for training, {} Y for validation.".format(n_train, n_test))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("using {} device.".format(device))
    
   
    net = mynet()
    net = nn.DataParallel(net)
    # PATH_checkpoint = "/p300/PhasedArray/Y_5_20_20_20/mynet_v1changed2_8.26_BS128_SGD_changelr_sdshuffleloss_losscos.pth"
    # net.load_state_dict(torch.load(PATH_checkpoint))
    net.to(device)
    # print(str(net))
    save_path = './mynetv1_9.24_BS32_OptimizerSGD_LRCustomizedPlateau0.01_ActivationFuncReLU_LossFuncCos_Node09_500epoch_withoutPositionInfo.pth'
    for p in net.parameters():
        p.requires_grad = True
        # print(p, p.requires_grad)


    # criterion = nn.L1Loss()
    # criterion = nn.MSELoss()
    criterion = Custom_LossFunction()


    EPOCH = 500
    best_distance_train = 2.0
    train_steps = len(train_loader) # train_steps: how many iteration in one training epoch (iteration = ALLTrainData / Epoch)
    val_steps = len(valid_loader) # test_steps: how many iteration in one training epoch (iteration = ALLTrainData / Epoch)
    print("traning iteration {}, testing iteration {}.".format(train_steps, val_steps))

    since = time.time()

    
    # Initial learning rate
    lr = 0.01

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr)

    # Change learning rate based on specific need
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 150], gamma=0.5, verbose=True)
    # Stochastic Gradient Descent with warm restarts
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max= (EPOCH // 9) + 1) 
    # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 100,gamma = 0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    

    for epoch in range(EPOCH):

        # Change learning rate manually
        adjust_learning_rate(optimizer, epoch)
        print('learning rate: {:4f}'.format(lr))


        # Train
        net.train()

        running_loss = 0.0
        running_acc_distance_min = 0.0
        running_acc_distance_max = 0.0

        # Iterate over data
        train_bar = tqdm(train_loader)
        for step, (batch_Y, batch_X) in enumerate(train_bar):
            # Weight parameter gradient is cleared
            optimizer.zero_grad() 
            # Forward propagation & Backward propagation
            X_pred = net(batch_Y.to(device))
            # loss = criterion(X_pred, batch_X.to(device))
            loss, acc_distance_min, acc_distance_max = criterion(X_pred, batch_X.to(device))
            loss.backward()
            optimizer.step()
            # scheduler.step(running_loss)
            

            # Print statistics
            running_loss += loss.item()
            running_acc_distance_min += acc_distance_min.item()
            running_acc_distance_max += acc_distance_max.item()
            train_bar.desc = "train epoch[{}/{}]  loss:{:.5f}".format(
                epoch + 1,
                EPOCH,
                loss
            )

        scheduler.step(running_loss)
        # scheduler.step()
            
        writer.add_scalar('Train/Loss', running_loss / train_steps, epoch)
        writer.add_scalar('Train/acc_distance_min', running_acc_distance_min / train_steps, epoch)
        writer.add_scalar('Train/acc_distance_max', running_acc_distance_max / train_steps, epoch)
        
        print('[epoch %d] train_loss: %.5f  best_train_loss: %.5f  train_distance_min: %.5f  train_distance_max: %.5f ' %
            (epoch + 1, running_loss / train_steps, best_distance_train, running_acc_distance_min / train_steps, running_acc_distance_max / train_steps))

        
        # Validate
        if (epoch + 1) % 10 == 0:
            net.eval()

            val_running_loss = 0.0
            val_running_acc_distance_min = 0.0
            val_running_acc_distance_max = 0.0
            best_distance = 100000

            with torch.no_grad():
                val_bar = tqdm(valid_loader)
                for (val_Y, val_X) in val_bar:
                    X_pred = net(val_Y.to(device))
                    # val_loss = criterion(X_pred, val_X.to(device))
                    val_loss, val_acc_distance_min, val_acc_distance_max = criterion(X_pred, val_X.to(device))
                    val_running_loss += val_loss.item()
                    val_running_acc_distance_min += val_acc_distance_min.item()
                    val_running_acc_distance_max += val_acc_distance_max.item()
                    # val_running_acc += torch.eq(X_pred, val_X.to(device)).sum().item()
                    # val_running_acc += ((X_pred - val_X.to(device)) < 1e-3).float().mean().item()
                    

                    val_bar.desc = "valdt epoch[{}/{}] loss:{:.5f}".format(
                        epoch + 1,
                        EPOCH,
                        val_loss
                    )
                    
            scheduler.step(val_running_loss)
            # scheduler.step()

            writer.add_scalar('val/Loss', val_running_loss/ val_steps, epoch)
            writer.add_scalar('val/acc_distance_min', val_running_acc_distance_min / val_steps, epoch)
            writer.add_scalar('val/acc_distance_max', val_running_acc_distance_max / val_steps, epoch)  

            print('[epoch %d] train_loss: %.5f  train_distance_min: %.5f  train_distance_max: %.5f  val_loss: %.5f    test_distance_min: %.5f  test_distance_max: %.5f' %
                (epoch + 1, running_loss / train_steps, running_acc_distance_min / train_steps, running_acc_distance_max / train_steps, val_running_loss / val_steps, val_running_acc_distance_min / val_steps, val_running_acc_distance_max / val_steps))
            
                    
            if val_loss < best_distance:
                best_distance = val_loss
                torch.save(net.state_dict(), save_path)  # save parameters of model
                # torch.save(net, save_path) # save the whole model

    print('Finished Training')
    writer.flush()
    writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60 , time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_distance_train))

