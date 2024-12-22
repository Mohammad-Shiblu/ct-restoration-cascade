import torch
import torch.nn as nn
import torch.optim as optim
from models.unet import UNet
from train.base import BaseTrainer
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import lpips
import os
import matplotlib.pyplot as plt

class UNetTrainer(BaseTrainer):
    def __init__(self, config, logger):
        super().__init__(config, logger)
        self.criterion = lpips.LPIPS(net='alex').to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr = config['lr'])
        self.schedular = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', factor=0.5, patience=5)


    def setup_models(self):
        """Initialize the U-Net model"""
        self.model = UNet(in_channels=3, out_channels=3).to(self.device)

    def train_stage(self):
        best_val_loss = float('inf')
        save_dir = os.path.join(self.config['save_dir'], "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        patience_counter = 0
        PSNR_values = []
        ssim_values = []
        
        
        for epoch in range(self.config['epochs']):
            self.model.train()
            train_loss = 0.0
            train_psnr = 0.0
            train_ssim = 0.0
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['epochs']}")
            for inputs, clean_imgs in progress_bar:
                inputs, clean_imgs = inputs.to(self.device), clean_imgs.to(self.device)  

                for param in self.model.parameters():
                    param.requires_grad = True
                
                # forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                # compute loss
                loss = self.criterion(outputs, clean_imgs)

                # Ensure loss is a scalar
                if loss.dim() > 0:
                    loss = loss.mean()
                # Ensure loss requires gradients
                assert loss.requires_grad, "Loss tensor does not require gradients"
                
                loss.backward()
                self.optimizer.step()

                psnr, ssim_val = self.calculate_metrics(outputs, clean_imgs)
                train_loss += loss.item()
                train_psnr += psnr
                train_ssim += ssim_val

                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'psnr': f'{psnr:.4f}',
                    'ssim': f'{ssim_val:.4f}'
                })
            # callculate metrics for training
            train_loss /= len(self.train_loader)
            train_psnr /= len(self.train_loader)
            train_ssim /= len(self.train_loader)

            val_loss, val_psnr, val_ssim = self.validate_stage()

            self.scheduler.step(val_loss)

            # log metrics
            self.logger.info(
                f"Epoch {epoch+1} ------------------------------- "
                f"Train Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f} | "
                f"Val Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}"
            )

            # save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model_path = os.path.join(save_dir, "best_model.pth")
                dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_psnr': val_psnr,
                    'val_ssim': val_ssim,
                }
                torch.save(dict, model_path)
                self.logger.info(f"saved the best model with val loss. {val_loss:.4f}")
                


    def validate_stage(self):
        self.model.eval()
        val_loss = 0.0
        val_psnr = 0.0
        val_ssim = 0.0

        with torch.no_grad():
            for inputs, clean_imgs in self.val_loader:
                inputs, clean_imgs = inputs.to(self.device), clean_imgs.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, clean_imgs)
                if loss.dim() > 0:
                    loss = loss.mean()
                psnr, ssim_val = self.calculate_metrics(outputs, clean_imgs)
                val_loss += loss.item()
                val_psnr += psnr
                val_ssim += ssim_val

        val_loss /= len(self.val_loader)
        val_psnr /= len(self.val_loader)
        val_ssim /= len(self.val_loader)

        return val_loss, val_psnr, val_ssim
    
    def test_system(self):
        self.model.eval()
        test_psnr = 0.0
        test_ssim = 0.0

        with torch.no_grad():
            for inputs, clean_imgs in tqdm(self.test_loader, desc="Testing"):
                inputs, clean_imgs = inputs.to(self.device), clean_imgs.to(self.device)
                outputs = self.model(inputs)
                psnr, ssim_val = self.calculate_metrics(outputs, clean_imgs)
                test_psnr += psnr
                test_ssim += ssim_val
            
        test_psnr /= len(self.test_loader)
        test_ssim /= len(self.test_loader)

        self.logger.info(f"Test PSNR: {test_psnr:.2f}, SSIM: {test_ssim:.4f}")

        return test_psnr, test_ssim
    
    def visualize_results(self, num_samples=5):
        self.model.eval()

        noisy_imgs_list = []
        clean_imgs_list = []

        test_iter = iter(self.test_loader)
        for _ in range(num_samples):
            try:
                noisy_img, clean_img = next(test_iter)
            except StopIteration:
                # Reset the iterator if there are fewer samples than requested
                test_iter = iter(self.test_loader)
                noisy_img, clean_img = next(test_iter)
            noisy_imgs_list.append(noisy_img.to(self.device))
            clean_imgs_list.append(clean_img.to(self.device))
        
        noisy_imgs = torch.cat(noisy_imgs_list, dim=0)
        clean_imgs = torch.cat(clean_imgs_list, dim=0)
        with torch.no_grad():
            denoised_image = self.model(noisy_imgs)
        
        noisy_image = noisy_image[0].permute(1, 2, 0).cpu().numpy()
        denoised_image = denoised_image[0].permute(1, 2, 0).cpu().numpy()  
        clean_image = clean_image[0].permute(1, 2, 0).cpu().numpy()

        # plot the image   
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
        for i in range(num_samples):
            noisy_img = noisy_image[0].permute(1, 2, 0).cpu().numpy()
            noisy_img = self.normalize(noisy_img)
            axes[i, 0].imshow(noisy_img)
            axes[i, 0].set_title("Noisy Image")
            axes[i, 0].axis("off")
            
            denoised_img = denoised_image[0].permute(1, 2, 0).cpu().numpy()
            denoised_img = self.normalize(denoised_img)
            axes[i, 1].imshow(denoised_img)
            axes[i, 1].set_title("Denoised Image")
            axes[i, 1].axis("off")
            
            clean_img = clean_image[0].permute(1, 2, 0).cpu().numpy()
            clean_img = self.normalize(clean_img)
            axes[i, 2].imshow(clean_img)
            axes[i, 2].set_title("Clean Image")
            axes[i, 2].axis("off")

        plt.tight_layout()
        save_dir = os.path.join(self.config['save_dir'], "visualizations")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"denoising_results.png"))
        
        self.logger.info(f"Saved visualization for samples.")


    def run(self):
        self.logger.info("-------------------------Training------------------------")
        self.train_stage()
        
        self.logger.info("-------------------------Testing------------------------")
        self.test_system()

        self.logger.info("-----------------------Visualizing-----------------------")
        self.visualize_results(num_samples=5)




        

            
        