from src.losses.base import DistillationLoss
from src.losses.cross_entropy import CrossEntropy
from src.losses.feature_kd import FeatureKD
from src.losses.hinton import HintonKD
from src.losses.mgd import MGDLoss
from src.losses.relational import SimilarityPreservationLoss
from src.losses.aid_teacher_adaptation import AIDTeacherAdaptationLoss
from src.losses.lwdetr_small_loss import LWDETRLoss

# Дистилляция семантической сегментации
from src.losses.cwd import ChannelWiseKD
from src.losses.dist import DISTLoss
from src.losses.fitnets import FitNetsKD
from src.losses.pixel_kd import PixelWiseKD
from src.losses.yolov8n_loss import YOLOv8Loss
from src.losses.dckd_loss import DCKDLoss
from src.losses.kd_detr_loss import KDDETRLoss

__all__ = [
    "CrossEntropy",
    "DistillationLoss",
    "FeatureKD",
    "HintonKD",
    "MGDLoss",
    "SimilarityPreservationLoss",
    "AIDTeacherAdaptationLoss",
    "LWDETRLoss",
    "ChannelWiseKD",
    "DISTLoss",
    "FitNetsKD",
    "PixelWiseKD",
    "YOLOv8Loss",
    "DCKDLoss",
    "KDDETRLoss",
]
