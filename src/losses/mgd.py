import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.base import DistillationLoss

class MGDLoss(DistillationLoss):
    requires_teacher = True
    def __init__(self, layer_name: str, s_channels: int, t_channels: int, lambda_mgd: float = 1.0, mask_ratio: float = 0.5):
        super().__init__()
        self.layer_name = layer_name
        self.lambda_mgd = lambda_mgd
        self.mask_ratio = mask_ratio
        self.required_features = (layer_name,)
        self.generation_config = nn.Sequential(
            nn.Conv2d(s_channels, t_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(t_channels, t_channels, kernel_size=1)
        )

    def forward(self, student_logits, teacher_logits, labels, student_features=None, teacher_features=None, **kwargs):
        if student_features is None or teacher_features is None:
            raise TypeError("MGDLoss: Trainer не передал признаки.")
        s_feat = student_features[self.layer_name]
        t_feat = teacher_features[self.layer_name]
        device = s_feat.device
        n, c, h, w = s_feat.shape
        mask = torch.bernoulli(torch.full((n, 1, h, w), 1 - self.mask_ratio)).to(device)
        out = self.generation_config(s_feat * mask)
        mgd_loss = F.mse_loss(out, t_feat.detach(), reduction='sum') / n
        ce_loss = F.cross_entropy(student_logits, labels)
        total_loss = ce_loss + (mgd_loss * self.lambda_mgd)
        return {"total": total_loss, "ce": ce_loss, "mgd": mgd_loss}
