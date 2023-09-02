import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torch.utils.data import DataLoader
import wandb

from dataset import KITTIDataset
from model import VORefinementLSTM


def avg_dict(all_metrics):
    keys = all_metrics[0].keys()
    avg_metrics = {}
    for key in keys:
        avg_metrics[key] = np.mean([all_metrics[i][key].cpu().detach().numpy() for i in range(len(all_metrics))])
    return avg_metrics





def _load_ckpnt(args,model,optimizer):
        ckpnt = torch.load(args.ckpnt)
        model.load_state_dict(ckpnt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpnt["optimizer_state_dict"])
        start_epoch = ckpnt["epoch"]
        val_acc_prev_best = ckpnt['best_loss']
        return start_epoch, val_acc_prev_best





def main():

    parser = argparse.ArgumentParser(description='Load Dataset')
    parser.add_argument('--data_path', type=str, default='../KITTI/sequences/') 
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_wandb', default = False)
    parser.add_argument('--eval_interval', type=int, default = 5)
    parser.add_argument('--ckpnt', type=str, default="Model_final")
    parser.add_argument('--gt', type=str, default='../KITTI/poses/') 
    
    args = parser.parse_args()
    train_dataset = KITTIDataset( image_dir = args.data_path, image_filename_pattern="{}.png" ,length=224, width = 224, odom_file= '../KITTI/odoms', gt_file = args.gt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader = DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = 4)
    
    val_loader = DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = 4)
    
    model = VORefinementLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
        
    if (args.use_wandb):
        wandb.init(project="Vo Refinement")
        
    train_loss_prev_best = float("inf")
    if args.ckpnt is None:
        args.ckpnt = "model.pt"    
        
    if os.path.exists(args.ckpnt):
            start_epoch, val_acc_prev_best = _load_ckpnt(args,model,optimizer)
            
            
    MSE_loss = nn.MSELoss(reduction='mean')
    
    for epoch in range(args.epochs):
        
        print('epoch', epoch)
        
        model.train()
        train_metrics_list = []
        i = 0
        train_loss = 0
        for img, odom, gt in train_loader:
            img, odom, gt = img.to(device), odom.to(device), gt.to(device)
            new_odom = model(img,odom)
            loss = (100*MSE_loss(new_odom[:,:,:3],gt[:,:,:3]) + MSE_loss(new_odom[:,:,3:],gt[:,:,3:]) )
            if args.use_wandb:
                wandb.log({"loss" :loss.item()})
            train_loss += loss.item()
            _metric = OrderedDict(recon_loss=loss.item())
            train_metrics_list.append(_metric)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            i+=1  
            
        lr_scheduler.step()
        train_metrics = avg_dict(train_metrics_list) 
        print(epoch, train_loss/8.)
               
        if args.use_wandb:
                wandb.log({"loss" :train_loss/8.})
                
        if (epoch)%(args.eval_interval) == 0:
            torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, 'checkpoint.pth')
            
            train_loss = train_metrics['recon_loss']  
            
            if train_loss <= train_loss_prev_best:
                print("Saving Checkpoint")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": train_loss
                }, args.ckpnt)
                train_loss_prev_best = train_loss
            else:
                print("Updating Checkpoint")
                checkpoint = torch.load(args.ckpnt)
                checkpoint["epoch"] += 1
                torch.save(checkpoint, args.ckpnt)   
    
       
        #Validation
        
        if (epoch)%(args.eval_interval) == 0:
            model.eval()
            val_metrics_list = []
            val_loss = 0
            with torch.no_grad():
                i = 0
                for img, odom, gt in val_loader:
                    img, odom, gt = img.to(device), odom.to(device), gt.to(device)
                    new_odom = model(img,odom)
                    loss = (MSE_loss(new_odom[:,:,:3],gt[:,:,:3]) + 100*MSE_loss(new_odom[:,:,3:],gt[:,:,3:]) )
                    train_loss += loss.item()
                    _metric = OrderedDict(recon_loss=loss.item())
                    val_metrics_list.append(_metric)
                    
                    if i == 0:
                        gt = gt.cpu().numpy()
                        new_odom = new_odom.cpu().numpy()
                        odom = odom.cpu().numpy()
                        plt.plot(new_odom[i,:,0], new_odom[i,:,2])
                        plt.plot(gt[i,:,0], gt[i,:,2])
                        plt.plot(odom[i,:,0], odom[i,:,2])
                        plt.savefig("{}_gt".format(epoch))
                        plt.close()
                    i+=1
                    
            val_metrics = avg_dict(val_metrics_list)
            print("Val Metrics:")
            print(val_metrics)
            if args.use_wandb:
                wandb.log(val_metrics)    



if __name__ == '__main__':
    main()
