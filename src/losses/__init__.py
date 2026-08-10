from src.losses.base import DistillationLoss
from src.losses.cross_entropy import CrossEntropy
from src.losses.feature_kd import FeatureKD
from src.losses.hinton import HintonKD
from src.losses.mgd import MGDLoss
from src.losses.relational import SimilarityPreservationLoss
from src.losses.aid_teacher_adaptation import AIDTeacherAdaptationLoss
from src.losses.lwdetr_small_loss import LWDETRLoss
from src.losses.faster_rcnn_loss import FasterRCNNLoss

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
]
