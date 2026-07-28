import torch
import torch.nn as nn
import torch.nn.functional as F
from src.losses.base import DistillationLoss

class SimilarityPreservationLoss(DistillationLoss):
    requires_teacher = True

    def __init__(self, lambda_sp: float = 1000.0):
        super().__init__()
        self.lambda_sp = lambda_sp
        self.required_features = ('stages.3',)

    def forward(self, student_logits, teacher_logits, labels, student_features=None, teacher_features=None, **kwargs):
        if student_features is None or teacher_features is None:
            raise TypeError("SP Loss: Trainer не передал признаки. Проверьте required_features.")

        s = student_features['stages.3']
        t = teacher_features['stages.3']

        if s.dim() == 3: s = s.mean(dim=1)
        if t.dim() == 3: t = t.mean(dim=1)

        def get_similarity_matrix(f):
            f = f.view(f.size(0), -1)
            f = F.normalize(f, p=2, dim=1)
            return torch.mm(f, f.t())

        G_s = get_similarity_matrix(s)
        G_t = get_similarity_matrix(t)

        sp_loss = F.mse_loss(G_s, G_t.detach())
        ce_loss = F.cross_entropy(student_logits, labels)

        return {
            "total": ce_loss + (sp_loss * self.lambda_sp),
            "ce": ce_loss,
            "sp_distill": sp_loss
        }
