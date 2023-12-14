import torch
import torch.nn as nn

from eval import eval_knn
import utils

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: torch.device,
        train_dataloader_aug: torch.utils.data.DataLoader,
        train_dataloader_plain: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        config: dict
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader_aug
        self.train_dataloader_plain = train_dataloader_plain
        self.test_dataloader = test_dataloader
        self.config = config
        self.device = device
                
        self.print_config()
        
    def print_config(self):
        print("encoder_type:", self.config.encoder_type)
        print("epochs:", self.config.epochs)
        print("batch_size:", self.config.batch_size)
        
            
    def train_one_epoch(self, epoch):
        print(f"\nEpoch: {epoch}")
        self.model.train()
        
        sum_loss = 0
        for batch_idx, (crops, target) in enumerate(self.train_dataloader):
            crops = [crop.to(self.device) for crop in crops]
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            student_out, teacher_out = self.model(crops, training=True)
            loss = self.loss_fn(student_out, teacher_out)
            sum_loss += loss
            loss.backward()
            utils.clip_gradients(self.model)
            self.optimizer.step()
            
            self.model.update_teacher(self.config.teacher_momentum)
            
            
            self.print_progress(batch_idx, sum_loss, len(crops[0]))
            

    def print_progress(self, batch_idx, sum_loss, crop_len):
            total_samples_processed = (batch_idx + 1) * crop_len
            total_samples = len(self.train_dataloader.dataset)
            percentage = 100. * total_samples_processed / total_samples
            if batch_idx % 3 == 0:
                # print(f"[{total_samples_processed}/{total_samples} ({percentage:.0f}%)]  Loss: {loss.item():.6f}\r", end='', flush=True)
                print(f"[{total_samples_processed}/{total_samples} ({percentage:.0f}%)]  Loss: {(sum_loss/(batch_idx+1)):.6f}\r", end='', flush=True)
      
    
    def train(self):
        self.eval()
        for epoch in range(self.config.epochs):
            self.train_one_epoch(epoch)
            print()
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            self.eval()
            self.save_model()
            self.log_metrics()
            
    
    def eval(self):
        knn_acc = eval_knn(self.model, self.train_dataloader_plain, self.test_dataloader, self.device)
        
        print(f"\nKNN Accuracy: {knn_acc:.4f}")
    
    def save_model(self):
        pass
    
    def log_metrics(self):
        pass
        
        
        
        
