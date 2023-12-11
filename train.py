import torch
import torchvision
import numpy as np
import argparse
import yaml

from utils import get_train_test_dataloaders, Config
from models.DINO import DINO, DINO_Loss
from Trainer import Trainer

def train(config: dict):
    # Set seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    train_dataloader, test_dataloader = get_train_test_dataloaders(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DINO(enc_type=config.encoder_type, out_dim=config.out_dim)
    model = model.to(device)
    
    lr = 0.0005 * config.batch_size / 256
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # TODO : Implement scheduler
    scheduler = None
    
    loss_fn = DINO_Loss(config.out_dim).to(device)
    
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        config=config
    )
    
    trainer.train()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "DINO");

    parser.add_argument('--config', type=str, default=None, help='Config YAML file')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        yml_dict = yaml.load(f, Loader=yaml.FullLoader)
    
    config = Config(yml_dict)
    
    train(config)
    
    
    