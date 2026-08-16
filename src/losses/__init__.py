from src.losses.base import DistillationLoss
from src.losses.cross_entropy import CrossEntropy
from src.losses.feature_kd import FeatureKD
from src.losses.hinton import HintonKD
from src.losses.mgd import MGDLoss
from src.losses.relational import SimilarityPreservationLoss
from src.losses.aid_teacher_adaptation import AIDTeacherAdaptationLoss
from src.losses.lwdetr_small_loss import LWDETRLoss
from src.losses.faster_rcnn_loss import FasterRCNNLoss

# Дистилляция семантической сегментации
from src.losses.cwd import ChannelWiseKD
from src.losses.dist import DISTLoss
from src.losses.fitnets import FitNetsKD
from src.losses.pixel_kd import PixelWiseKD
from src.losses.yolo_loss import YOLO
from src.losses.dckd_loss import DCKDLoss
from src.losses.kd_detr_loss import KDDETRLoss
from src.losses.clockdistill_loss import CLoCKDistillLoss

__all__ = [
    "CrossEntropy",
    "DistillationLoss",
    "FeatureKD",
    "HintonKD",
    "MGDLoss",
    "SimilarityPreservationLoss",
    "AIDTeacherAdaptationLoss",
    "LWDETRLoss",
    "FasterRCNNLoss"
    "ChannelWiseKD",
    "DISTLoss",
    "FitNetsKD",
    "PixelWiseKD",
    "YOLO",
    "DCKDLoss",
    "KDDETRLoss",
    "CLoCKDistillLoss",
]
