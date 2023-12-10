import torch
import torch.nn as nn



class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        config: dict
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        
    def train_one_epoch(self, epoch):
        pass
    
    def train(self):
        for epoch in range(self.config.epochs):
            self.train_one_epoch(epoch)
            
    
    def eval(self):
        pass
    
    def save_model(self):
        pass
    
    def log_metrics(self):
        pass
        
        
        
        
