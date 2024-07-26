import torch
from diffusers import schedulers
from typing import Union
from enum import Enum, auto

class Datatypes(Enum):
    FLOAT16 = auto()
    FLOAT32 = auto()
    FLOAT64 = auto()

    def to_torch_dtype(self):
        if self == Datatypes.FLOAT16:
            return torch.float16
        elif self == Datatypes.FLOAT32:
            return torch.float32
        elif self == Datatypes.FLOAT64:
            return torch.float64
        else:
            raise ValueError(f"Unsupported datatype: {self}")

scheduler_types = Union [
    schedulers.DDPMScheduler,
    schedulers.DPMSolverMultistepScheduler,
    schedulers.DPMSolverSinglestepScheduler,
    schedulers.EDMDPMSolverMultistepScheduler,
    schedulers.EDMEulerScheduler,
    schedulers.EulerAncestralDiscreteScheduler,
    schedulers.EulerDiscreteScheduler,
    schedulers.HeunDiscreteScheduler,
    schedulers.IPNDMScheduler,
    schedulers.KDPM2AncestralDiscreteScheduler,
    schedulers.KDPM2DiscreteScheduler
]

class DiffusionConfig(Enum):
    BETA_END = 0.012
    BETA_START = 0.00085
    NUM_TRAINING_STEPS = 1000