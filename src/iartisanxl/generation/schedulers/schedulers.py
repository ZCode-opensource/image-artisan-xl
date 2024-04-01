import attr
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSDEScheduler,
    DPMSolverSinglestepScheduler,
    EDMDPMSolverMultistepScheduler,
    EDMEulerScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LCMScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
)


@attr.s
class Scheduler:
    name = attr.ib()
    scheduler_class = attr.ib()
    scheduler_args = attr.ib()


schedulers = [
    Scheduler("DDIM", DDIMScheduler, {}),
    Scheduler("DDPM", DDPMScheduler, {}),
    Scheduler("DEIS", DEISMultistepScheduler, {}),
    Scheduler("DPM++ 2M", DPMSolverMultistepScheduler, {"use_karras_sigmas": False}),
    Scheduler("DPM++ 2M Karras", DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    Scheduler(
        "DPM++ 2M SDE", DPMSolverMultistepScheduler, {"use_karras_sigmas": False, "algorithm_type": "sde-dpmsolver++"}
    ),
    Scheduler(
        "DPM++ 2M SDE Karras",
        DPMSolverMultistepScheduler,
        {"use_karras_sigmas": True, "algorithm_type": "sde-dpmsolver++"},
    ),
    Scheduler("DPM++ 2S", DPMSolverSinglestepScheduler, {"use_karras_sigmas": False}),
    Scheduler("DPM++ 2S Karras", DPMSolverSinglestepScheduler, {"use_karras_sigmas": True}),
    Scheduler("DPM++ SDE", DPMSolverSDEScheduler, {"use_karras_sigmas": False, "noise_sampler_seed": 0}),
    Scheduler("DPM++ SDE Karras", DPMSolverSDEScheduler, {"use_karras_sigmas": True, "noise_sampler_seed": 0}),
    Scheduler("DPM++ 2M EDM", EDMDPMSolverMultistepScheduler, {}),
    Scheduler("DPM++ 2M EDM Karras", EDMDPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
    Scheduler("Euler", EulerDiscreteScheduler, {}),
    Scheduler("Euler Ancestral", EulerAncestralDiscreteScheduler, {}),
    Scheduler("Euler EDM", EDMEulerScheduler, {}),
    Scheduler("Euler EDM Karras", EDMEulerScheduler, {"use_karras_sigmas": True}),
    Scheduler("Euler Karras", EulerDiscreteScheduler, {"use_karras_sigmas": True}),
    Scheduler("Heun", HeunDiscreteScheduler, {}),
    Scheduler("Heun Karras", HeunDiscreteScheduler, {"use_karras_sigmas": True}),
    Scheduler("KDPM 2", KDPM2DiscreteScheduler, {}),
    Scheduler("KDPM 2 Ancestral", KDPM2AncestralDiscreteScheduler, {}),
    Scheduler("LCM", LCMScheduler, {}),
    Scheduler("LMS", LMSDiscreteScheduler, {}),
    Scheduler("LMS Karras", LMSDiscreteScheduler, {"use_karras_sigmas": True}),
    Scheduler("PNDM", PNDMScheduler, {}),
    Scheduler("UniPC", UniPCMultistepScheduler, {}),
    Scheduler("UniPC Karras", UniPCMultistepScheduler, {"use_karras_sigmas": True}),
    Scheduler("Turbo - Euler", EulerDiscreteScheduler, {}),
    Scheduler("Turbo - Euler Ancestral", EulerAncestralDiscreteScheduler, {}),
    Scheduler("Turbo - Euler Karras", EulerDiscreteScheduler, {"use_karras_sigmas": True}),
]
