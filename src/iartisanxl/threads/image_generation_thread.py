import time
import logging

import torch
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.generation.generation_data_object import ImageGenData


# pylint: disable=no-member
class ImageGenerationThread(QThread):
    status_changed = pyqtSignal(str)
    progress_update = pyqtSignal(int, torch.Tensor)
    generation_finished = pyqtSignal(Image.Image, float)
    generation_error = pyqtSignal(str, bool)
    generation_aborted = pyqtSignal()

    def __init__(self, base_pipeline, rendering_generation_data: ImageGenData):
        super().__init__()
        self.logger = logging.getLogger()
        self.base_pipeline = base_pipeline
        self.image_generation_data = rendering_generation_data
        self.start_time = None

    def run(self):
        kwargs = {}
        kwargs["callback_steps"] = 1
        kwargs["callback"] = self.step_progress_update

        if len(self.image_generation_data.positive_prompt_clipl) > 0:
            kwargs["prompt_2"] = self.image_generation_data.positive_prompt_clipl

        if len(self.image_generation_data.negative_prompt_clipl) > 0:
            kwargs[
                "negative_prompt_2"
            ] = self.image_generation_data.negative_prompt_clipl

        size = (
            self.image_generation_data.image_height,
            self.image_generation_data.image_width,
        )

        self.status_changed.emit("Generating image...")

        self.start_time = time.time()

        try:
            image = self.base_pipeline(
                original_size=size,
                target_size=size,
                prompt=self.image_generation_data.positive_prompt_clipg,
                seed=self.image_generation_data.seed,
                height=self.image_generation_data.image_height,
                width=self.image_generation_data.image_width,
                on_aborted_function=self.on_abort,
                status_update=self.on_status_changed,
                num_inference_steps=self.image_generation_data.steps,
                guidance_scale=self.image_generation_data.guidance,
                negative_original_size=(512, 512),
                negative_target_size=size,
                negative_prompt=self.image_generation_data.negative_prompt_clipg,
                cross_attention_kwargs={"scale": self.image_generation_data.lora_scale},
                clip_skip=self.image_generation_data.clip_skip,
                **kwargs,
            )
        except NotImplementedError as e:
            self.generation_error.emit(
                "There was an error trying to generate the image.", True
            )
            self.logger.error(
                "NotImplementedError: there was an error trying to generate the image with this message: %s",
                e,
            )
            self.logger.debug("NotImplementedError exception occurred", exc_info=True)
            return
        except RuntimeError as e:
            self.generation_error.emit(
                "There was an error trying to generate the image.", True
            )
            self.logger.error(
                "RuntimeError: there was an error trying to generate the image with this message: %s",
                e,
            )
            self.logger.debug("RuntimeError exception occurred", exc_info=True)
            return

        if image is not None:
            end_time = time.time()
            duration = end_time - self.start_time
            self.generation_finished.emit(image, duration)

    def on_status_changed(self, message: str):
        self.status_changed.emit(message)

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)

    def abort_generation(self):
        self.base_pipeline.abort = True

    def on_abort(self):
        self.base_pipeline.abort = False
        self.generation_aborted.emit()
