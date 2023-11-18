import logging

import torch

from diffusers import AutoencoderKL, ControlNetModel
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.pipelines.txt_pipeline import ImageArtisanTextPipeline
from iartisanxl.pipelines.controlnet_txt_pipeline import (
    ImageArtisanControlNetTextPipeline,
)
from iartisanxl.generation.generation_data_object import ImageGenData
from iartisanxl.generation.schedulers.schedulers_utils import load_scheduler
from iartisanxl.nodes.stable_difussion_xl_node import StableDiffusionXLModelNode


class PipelineSetupThread(QThread):
    status_changed = pyqtSignal(str)
    pipeline_ready = pyqtSignal(object)
    pipeline_error = pyqtSignal(str)
    pipeline_abort = pyqtSignal()

    def __init__(
        self,
        rendering_generation_data: ImageGenData,
        model_offload: bool = False,
        sequential_offload: bool = False,
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()

        self.logger = logging.getLogger()
        self.image_generation_data = rendering_generation_data
        self.model_offload = model_offload
        self.sequential_offload = sequential_offload
        self.torch_dtype = torch_dtype
        self.abort = False

    # pylint: disable=no-member
    def run(self):
        self.status_changed.emit("Loading model")
        pipeline = None
        vae = None

        try:
            if self.image_generation_data.vae is not None:
                model_node = StableDiffusionXLModelNode(
                    path=self.image_generation_data.model.path,
                    torch_dtype=self.torch_dtype,
                )
                (
                    text_encoder_1,
                    text_encoder_2,
                    tokenizer_1,
                    tokenizer_2,
                    unet,
                ) = model_node()

                if (
                    len(self.image_generation_data.vae.path) > 0
                    or self.image_generation_data.model.type == "safetensors"
                ):
                    self.status_changed.emit("Loading vae...")
                    vae = AutoencoderKL.from_pretrained(
                        self.image_generation_data.vae.path,
                        torch_dtype=self.torch_dtype,
                    )
                else:
                    vae = AutoencoderKL.from_pretrained(
                        self.image_generation_data.model.path,
                        torch_dtype=self.torch_dtype,
                    )

                scheduler = load_scheduler(self.image_generation_data.base_scheduler)

                if len(self.image_generation_data.controlnets) > 0:
                    controlnets = []

                    for controlnet in self.image_generation_data.controlnets:
                        controlnet_model = ControlNetModel.from_pretrained(
                            controlnet.model_path,
                            torch_dtype=torch.float16,
                            use_safetensors=True,
                            variant="fp16",
                        )
                        controlnets.append(controlnet_model)

                    pipeline = ImageArtisanControlNetTextPipeline(
                        vae=vae,
                        text_encoder=text_encoder_1,
                        text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer_1,
                        tokenizer_2=tokenizer_2,
                        unet=unet,
                        scheduler=scheduler,
                        controlnet=controlnets,
                    )
                else:
                    pipeline = ImageArtisanTextPipeline(
                        vae=vae,
                        text_encoder=text_encoder_1,
                        text_encoder_2=text_encoder_2,
                        tokenizer=tokenizer_1,
                        tokenizer_2=tokenizer_2,
                        unet=unet,
                        scheduler=scheduler,
                    )

                if not self.abort:
                    if self.sequential_offload:
                        self.logger.debug("Sequential cpu offload enabled")
                        pipeline.enable_sequential_cpu_offload()
                        pipeline.sequential_cpu_offloaded = True
                    else:
                        if self.model_offload:
                            self.logger.debug("Model cpu offload enabled")
                            pipeline.enable_model_cpu_offload()
                            pipeline.model_cpu_offloaded = True
                        else:
                            pipeline.to("cuda")

                self.pipeline_ready.emit(pipeline)
        except EnvironmentError as env_error:
            self.logger.error(
                "Error loading the model with this message: %s", str(env_error)
            )
            self.pipeline_error.emit(
                "There was a problem loading the model. Try to select and load it again."
            )
