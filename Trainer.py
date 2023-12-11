import torch
import torch.nn as nn


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        device: torch.device,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        config: dict
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.config = config
        self.device = device
            
    def train_one_epoch(self, epoch):
        print(f"\nEpoch: {epoch}")
        for batch_idx, (crops, target) in enumerate(self.train_dataloader):
            crops = [crop.to(self.device) for crop in crops]
            target = target.to(self.device)
            
            self.optimizer.zero_grad()
            
            student_out, teacher_out = self.model(crops, training=True)
            loss = self.loss_fn(student_out, teacher_out)
            
            loss.backward()
            # TODO: clip gradients
            self.optimizer.step()
            
            self.model.update_teacher(self.config.teacher_momentum)
            
            self.print_progress(batch_idx, loss, len(crops[0]))
            

    def print_progress(self, batch_idx, loss, crop_len):
            total_samples_processed = (batch_idx + 1) * crop_len
            total_samples = len(self.train_dataloader.dataset)
            percentage = 100. * total_samples_processed / total_samples
            if batch_idx % 3 == 0:
                print(f"[{total_samples_processed}/{total_samples} ({percentage:.0f}%)]  Loss: {loss.item():.6f}\r", end='', flush=True)
      
    
    def train(self):
        for epoch in range(self.config.epochs):
            self.train_one_epoch(epoch)
            
            if self.scheduler is not None:
                self.scheduler.step()
                
            self.eval()
            self.save_model()
            self.log_metrics()
            
    
    def eval(self):
        pass
    
    def save_model(self):
        pass
    
    def log_metrics(self):
        pass
        
        
        
        
