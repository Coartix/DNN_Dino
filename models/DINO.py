import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class DINO_Head(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, bottleneck_dim):
        super(DINO_Head, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim)
        )
        
        self.apply(self._init_weights)
        
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, p=2, dim=-1)
        
        return self.last_layer(x)
    
class DINO_ViT(nn.Module):
    def __init__(self):
        super(DINO_ViT, self).__init__()
        
        # TODO : Implement DINO with teacher and student networks as ViT_small and DINOHead on top
        pass
    

class DINO(nn.Module):
    def __init__(self, enc_type, enc_out_dim=1000, out_dim=128, hidden_dim=2048, bottleneck_dim=256):
        super(DINO, self).__init__()
        
        if enc_type == "resnet18":
            enc = torchvision.models.resnet18
        elif enc_type == "resnet34":
            enc = torchvision.models.resnet34
        elif enc_type == "vit":
            # TODO : Implement ViT
            pass
        
        self.student_backbone = enc()
        self.student_head = DINO_Head(enc_out_dim, out_dim, hidden_dim, bottleneck_dim)
        
        self.teacher_backbone = enc()
        self.teacher_head = DINO_Head(enc_out_dim, out_dim, hidden_dim, bottleneck_dim)
        
        self._init_freeze_teacher()

    def _init_freeze_teacher(self):
        # init teacher weights with student weights
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        
        # freeze teacher weights
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
            
        for param in self.teacher_head.parameters():
            param.requires_grad = False
            
    def update_teacher(self, teacher_momentum):
        for (student_ps_backbone, teacher_ps_backbone), (student_ps_head, teacher_ps_head) in zip(
            zip(self.student_backbone.parameters(), self.teacher_backbone.parameters()),
            zip(self.student_head.parameters(), self.teacher_head.parameters())
        ):
            teacher_ps_backbone.data = teacher_ps_backbone.data * (1-teacher_momentum) + student_ps_backbone.data * teacher_momentum
            teacher_ps_head.data = teacher_ps_head.data * (1-teacher_momentum) + student_ps_head.data * teacher_momentum

            
    def _student_forward(self, x):
        x = self.student_backbone(x)
        x = self.student_head(x)
        
        return x
    
    def _teacher_forward(self, x):
        x = self.teacher_backbone(x)
        x = self.teacher_head(x)
        
        return x
        
    def forward(self, x, training=False):
        if not training: return self._student_forward(x)
     
        all_crops = torch.cat(x, dim=0)
        global_crops = torch.cat(x[:2], dim=0)
  
        student_out = self._student_forward(all_crops)
        teacher_out = self._teacher_forward(global_crops)
        
        student_out = student_out.chunk(len(x))
        teacher_out = teacher_out.chunk(2)
        
        return student_out, teacher_out

class DINO_Loss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super(DINO_Loss, self).__init__()
        
        self.out_dim = out_dim
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        
        self.register_buffer("center", torch.zeros(1, out_dim))
        
    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_center = torch.cat(teacher_out).mean(dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
        
    def forward(self, _student_out, _teacher_out):
        student_out = [s / self.student_temp for s in _student_out]
        teacher_out = [F.softmax((t - self.center) / self.teacher_temp, dim=-1).detach() for t in _teacher_out]
                
        loss = 0
        count = 0
        
        for t_ix, t in enumerate(teacher_out):
            for s_ix, s in enumerate(student_out):
                if t_ix == s_ix:
                    continue
                
                tmp_loss = torch.sum(-t * F.log_softmax(s, dim=-1), dim=-1)
                loss += tmp_loss.mean()
                count += 1
        
        loss /= count
        self.update_center(student_out)

        return loss
