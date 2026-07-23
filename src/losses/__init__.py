from src.losses.base import DistillationLoss
from src.losses.cross_entropy import CrossEntropy
from src.losses.feature_kd import FeatureKD
from src.losses.hinton import HintonKD
from src.losses.mgd import MGDLoss

__all__ = ["CrossEntropy", "DistillationLoss", "FeatureKD", "HintonKD", "MGDLoss"]
