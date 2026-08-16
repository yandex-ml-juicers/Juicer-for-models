from src.losses.base import DistillationLoss
from src.losses.composite import CompositeLoss
from src.losses.cross_entropy import CrossEntropy
from src.losses.feature_kd import FeatureKD
from src.losses.hinton import HintonKD
from src.losses.mgd import MGDLoss
from src.losses.relational import SimilarityPreservationLoss
from src.losses.aid_teacher_adaptation import AIDTeacherAdaptationLoss
from src.losses.lwdetr_small_loss import LWDETRLoss

# Дистилляция семантической сегментации
from src.losses.bpkd import BPKDLoss
from src.losses.cwd import ChannelWiseKD
from src.losses.dist import DISTLoss
from src.losses.distillation_only import DistillationOnlyLoss
from src.losses.fitnets import FitNetsKD
from src.losses.heteroakd import HeteroAKDLoss
from src.losses.pixel_kd import PixelWiseKD
from src.losses.yolo_loss import YOLO
from src.losses.dckd_loss import DCKDLoss
from src.losses.kd_detr_loss import KDDETRLoss
from src.losses.clockdistill_loss import CLoCKDistillLoss

# Лоссы самой сегментации (учитель не нужен)
from src.losses.dice import DiceLoss
from src.losses.focal import FocalLoss
from src.losses.lovasz import LovaszSoftmax
from src.losses.ohem import OhemCrossEntropy

__all__ = [
    "CompositeLoss",
    "CrossEntropy",
    "DistillationLoss",
    "FeatureKD",
    "HintonKD",
    "MGDLoss",
    "SimilarityPreservationLoss",
    "AIDTeacherAdaptationLoss",
    "LWDETRLoss",
    "BPKDLoss",
    "ChannelWiseKD",
    "DISTLoss",
    "DiceLoss",
    "DistillationOnlyLoss",
    "FitNetsKD",
    "FocalLoss",
    "HeteroAKDLoss",
    "LovaszSoftmax",
    "OhemCrossEntropy",
    "PixelWiseKD",
    "YOLO",
    "DCKDLoss",
    "KDDETRLoss",
    "CLoCKDistillLoss",
]
