"""Section 3: grasping and bite acquisition. See README.md in this folder."""

from .acquisition import acquire_cup, acquire_pinch, acquire_with_utensil
from .models import AnyGraspModel, ContactGraspNetModel
from .pca import pca_topdown_grasp
from .scoop import BowlTarget, ScoopConfig, ScoopPlan, plan_scoop
from .types import AcquisitionResult, GraspCandidate, GraspModel

__all__ = [
    "AcquisitionResult", "AnyGraspModel", "BowlTarget", "ContactGraspNetModel", "GraspCandidate", "GraspModel",
    "ScoopConfig", "ScoopPlan", "acquire_cup", "acquire_pinch", "acquire_with_utensil", "pca_topdown_grasp", "plan_scoop",
]
