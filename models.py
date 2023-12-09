import torch
import torch.nn as nn
import torch.nn.functional as F

class ViT_small(nn.Module):
    def __init__(self):
        super(ViT_small, self).__init__()
        
        # TODO : Implement ViT_small
        pass
    
class DINOHead(nn.Module):
    def __init__(self):
        super(DINOHead, self).__init__()
        
        # TODO : Implement DINOHead
        pass
    
class DINO(nn.Module):
    def __init__(self):
        super(DINO, self).__init__()
        
        # TODO : Implement DINO with teacher and student networks as ViT_small and DINOHead on top
        pass