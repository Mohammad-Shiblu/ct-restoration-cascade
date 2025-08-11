import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet
from train.base import BaseTrainer
from tqdm import tqdm
import lpips
import os
# os.environ['TORCH_HOME'] = '/home/woody/iwi5/iwi5240h/Masters-Thesis/pretrained/'
import matplotlib.pyplot as plt
from utils.metrics import compute_metrics, compute_ssim, compute_psnr
from utils.help import EarlyStopping

class UNetTrainer(BaseTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_data()
        self.setup_models()
        self.criterion = nn.L1Loss()  # lpips.LPIPS(net='alex').to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr = self.config['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor=0.5, patience=5)
        
    def setup_models(self):
        self.model = UNet(in_channels=1, out_channels=1).to(self.config['device'])

    def train(self):
        early_stopping = EarlyStopping(patience=self.config["patience"], min_delta=0.0)
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0.0
            
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for input, target in progress_bar:
                input, target = input.to(self.device), target.to(self.device)  

                # forward pass
                self.optimizer.zero_grad()
                output = self.model(input)
                
                # compute loss
                loss = self.criterion(output, target)

                # Ensure loss is a scalar
                if loss.dim() > 0:
                    loss = loss.mean()
                # Ensure loss requires gradients
                assert loss.requires_grad
                
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                })
            # callculate metrics for training
            train_loss /= len(self.train_loader)
            
            val_result = self.validate()

            self.scheduler.step(val_result[0])
            
            # save the loss per epoch for loss curves
            self.train_losses.append(train_loss)
            self.val_losses.append(val_result[0])
            
            # log metrics
            self.logger.info(
                f"Epoch {epoch+1} "
                f"Train Loss: {train_loss}, | "
                f"Val Loss: {val_result[0]:.4f}, PSNR: {val_result[1]:.4f}, SSIM: {val_result[2]:.4f}"
            )

            early_stopping.check_early_stop(val_result[0])
            if early_stopping.counter == 0:
                self.save_model()
                self.logger.info(f"best model has been saved at epoch {epoch} with val_loss of {val_result[0]}")

            if early_stopping.stop_training:
                self.logger.info(f"Training stopped due to early stopper at epoch {epoch}")
                self.plot_loss_curves()
                self.logger.info(f"Trained and val loss curves saved")
                break
        
        self.logger.info(f"All the epochs completed...")
        self.plot_loss_curves()
        self.logger.info(f"Trained and val loss curves saved")
                            
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0

        with torch.no_grad():
            for input, target in self.val_loader:
                input, target = input.to(self.device), target.to(self.device)
                
                output = self.model(input)
                loss = self.criterion(output, target)
                if loss.dim() > 0:
                    loss = loss.mean()
                psnr = compute_psnr(output, target, data_range=1.0)
                ssim = compute_ssim(output, target, data_range= 1.0)
                val_loss += loss.item()
                val_psnr += psnr
                val_ssim += ssim

        val_loss /= len(self.val_loader)
        val_psnr /= len(self.val_loader)
        val_ssim /= len(self.val_loader)

        return (val_loss, val_psnr, val_ssim)
            
    def test(self):
        self.model.eval()
        ori_psnr, ori_ssim, ori_rmse = 0, 0, 0
        pred_psnr, pred_ssim, pred_rmse = 0, 0, 0

        with torch.no_grad():
            for i, (input, target) in enumerate(self.test_loader):
                input, target = input.to(self.device), target.to(self.device)
                output = self.model(input)
                original, pred = compute_metrics(input, output, target, data_range=1.0)
                ori_rmse += original[0]
                ori_psnr += original[1]
                ori_ssim += original[2]

                pred_rmse += pred[0]
                pred_psnr += pred[1]
                pred_ssim += pred[2]
            
                self.save_image(i, input, output, target)

        ori_rmse /= len(self.test_loader)
        ori_psnr /= len(self.test_loader) 
        ori_ssim /= len(self.test_loader)
        pred_rmse /= len(self.test_loader)
        pred_psnr /= len(self.test_loader)
        pred_ssim /= len(self.test_loader)
                
        self.logger.info(f"Original PSNR: {ori_psnr:.4f}, SSIM: {ori_ssim:.4f}, RMSE: {ori_rmse:0.4f}")
        self.logger.info(f"After training model  Pred_PSNR: {pred_psnr:.4f}, PRED_SSIM: {pred_ssim:.4f}, PRED_RMSE: {pred_rmse:0.4f}")
    
    
    def run(self):
        self.logger.info("-------------------------Training------------------------")
        self.train()
        
        self.logger.info("-------------------------Testing------------------------")
        self.test()




        

            
        