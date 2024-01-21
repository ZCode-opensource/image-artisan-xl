import attr

from diffusers import (
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    HeunDiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    UniPCMultistepScheduler,
    DPMSolverSinglestepScheduler,
    DEISMultistepScheduler,
    DDPMScheduler,
    DPMSolverSDEScheduler,
    LCMScheduler,
)


@attr.s
class Scheduler:
    name = attr.ib()
    scheduler_class = attr.ib()
    scheduler_args = attr.ib()


schedulers = [
    Scheduler("DDIM", DDIMScheduler, dict()),
    Scheduler("DDPM", DDPMScheduler, dict()),
    Scheduler("DEIS", DEISMultistepScheduler, dict()),
    Scheduler("DPM++ 2M", DPMSolverMultistepScheduler, dict(use_karras_sigmas=False)),
    Scheduler("DPM++ 2M Karras", DPMSolverMultistepScheduler, dict(use_karras_sigmas=True)),
    Scheduler("DPM++ 2M SDE", DPMSolverMultistepScheduler, dict(use_karras_sigmas=False, algorithm_type="sde-dpmsolver++")),
    Scheduler("DPM++ 2M SDE Karras", DPMSolverMultistepScheduler, dict(use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")),
    Scheduler("DPM++ 2S", DPMSolverSinglestepScheduler, dict(use_karras_sigmas=False)),
    Scheduler("DPM++ 2S Karras", DPMSolverSinglestepScheduler, dict(use_karras_sigmas=True)),
    Scheduler("DPM++ SDE", DPMSolverSDEScheduler, dict(use_karras_sigmas=False, noise_sampler_seed=0)),
    Scheduler("DPM++ SDE Karras", DPMSolverSDEScheduler, dict(use_karras_sigmas=True, noise_sampler_seed=0)),
    Scheduler("Euler", EulerDiscreteScheduler, dict()),
    Scheduler("Euler Ancestral", EulerAncestralDiscreteScheduler, dict()),
    Scheduler("Euler Karras", EulerDiscreteScheduler, dict(use_karras_sigmas=True)),
    Scheduler("Heun", HeunDiscreteScheduler, dict()),
    Scheduler("Heun Karras", HeunDiscreteScheduler, dict(use_karras_sigmas=True)),
    Scheduler("KDPM 2", KDPM2DiscreteScheduler, dict()),
    Scheduler("KDPM 2 Ancestral", KDPM2AncestralDiscreteScheduler, dict()),
    Scheduler("LCM", LCMScheduler, dict()),
    Scheduler("LMS", LMSDiscreteScheduler, dict()),
    Scheduler("LMS Karras", LMSDiscreteScheduler, dict(use_karras_sigmas=True)),
    Scheduler("PNDM", PNDMScheduler, dict()),
    Scheduler("UniPC", UniPCMultistepScheduler, dict()),
    Scheduler("UniPC Karras", UniPCMultistepScheduler, dict(use_karras_sigmas=True)),
    Scheduler("Turbo - Euler", EulerDiscreteScheduler, dict()),
    Scheduler("Turbo - Euler Ancestral", EulerAncestralDiscreteScheduler, dict()),
    Scheduler("Turbo - Euler Karras", EulerDiscreteScheduler, dict(use_karras_sigmas=True)),
]
