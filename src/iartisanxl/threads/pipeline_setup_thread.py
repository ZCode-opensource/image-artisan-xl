import logging

import torch

from diffusers import AutoencoderKL
from huggingface_hub.utils._validators import HFValidationError
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.pipelines.txt_pipeline import ImageArtisanTextPipeline
from iartisanxl.generation.generation_data_object import ImageGenData
from iartisanxl.generation.schedulers.schedulers_utils import load_scheduler
from iartisanxl.modules.common.diffusers_utils import diffusers_models


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
    ):
        super().__init__()

        self.logger = logging.getLogger()
        self.image_generation_data = rendering_generation_data
        self.model_offload = model_offload
        self.sequential_offload = sequential_offload
        self.abort = False

    # pylint: disable=no-member
    def run(self):
        self.status_changed.emit("Loading model")
        pipeline = None
        vae = None

        try:
            if self.image_generation_data.vae is not None:
                if len(self.image_generation_data.vae.path) > 0:
                    self.status_changed.emit("Loading selected vae...")
                    vae = AutoencoderKL.from_pretrained(
                        self.image_generation_data.vae.path,
                        torch_dtype=torch.float16,
                    )

            if self.image_generation_data.model.type == "diffusers":
                self.status_changed.emit("Loading diffusers model...")
                pipeline = self.load_diffusers_pipeline(vae=vae)

                if pipeline is None:
                    self.pipeline_ready.emit(pipeline)
            else:
                try:
                    self.status_changed.emit("Loading safetensors model...")
                    pipeline = ImageArtisanTextPipeline.from_single_file(
                        self.image_generation_data.model.path,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        use_safetensors=True,
                        local_files_only=True,
                        original_config_file="./configs/sd_xl_base.yaml",
                        vae=vae,
                    )
                except FileNotFoundError as e:
                    self.pipeline_error.emit(f"{e}")
                    return

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
        except EnvironmentError:
            self.pipeline_error.emit(
                "There was a problem loading the model. Try to select and load it again."
            )

    def load_diffusers_pipeline(self, vae: AutoencoderKL = None):
        loaded_models = {}

        for model_info in diffusers_models:
            if self.abort:
                return
            else:
                try:
                    if model_info["name"] == "vae" and vae is not None:
                        loaded_models["vae"] = vae
                        continue

                    self.status_changed.emit(f"Loading {model_info['name']}...")
                    loaded_models[model_info["name"]] = model_info[
                        "model"
                    ].from_pretrained(
                        self.image_generation_data.model.path,
                        **model_info["args"],
                    )
                except HFValidationError:
                    self.pipeline_error.emit(
                        f"There was an error trying to load the {model_info['name']}. Try to select and load the model again."
                    )
                    return

        if self.abort:
            return
        else:
            self.status_changed.emit("Initializing the scheduler...")
            scheduler = load_scheduler(self.image_generation_data.base_scheduler)

        if self.abort:
            return
        else:
            self.status_changed.emit("Creating the pipeline...")
            pipeline = ImageArtisanTextPipeline(
                vae=loaded_models.get("vae"),
                text_encoder=loaded_models.get("text_encoder"),
                text_encoder_2=loaded_models.get("text_encoder_2"),
                tokenizer=loaded_models.get("tokenizer"),
                tokenizer_2=loaded_models.get("tokenizer_2"),
                unet=loaded_models.get("unet"),
                scheduler=scheduler,
            )

        return pipeline
