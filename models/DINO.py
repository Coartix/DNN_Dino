import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    def __init__(self):
        super(DINOHead, self).__init__()
        
        # TODO : Implement DINOHead
        pass
    
class DINO_ViT(nn.Module):
    def __init__(self):
        super(DINO_ViT, self).__init__()
        
        # TODO : Implement DINO with teacher and student networks as ViT_small and DINOHead on top
        pass
    

class DINO_ResNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super(DINO_ResNet, self).__init__()
        
        self.student_backbone = torchvision.models.resnet34()
        self.student_head = DINOHead()
        
        self.teacher_backbone = torchvision.models.resnet34()
        self.teacher_head = DINOHead()
        
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        global_crops, local_crops = crops[:, :2], crops[:, 2:]
        
            
