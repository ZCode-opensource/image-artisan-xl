import os
import random
import json
import logging
from typing import Optional
from datetime import datetime

import torch
import tomesd
from PIL import Image
from PyQt6.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QSpacerItem,
    QSizePolicy,
    QProgressBar,
)
from PyQt6.QtGui import QImage, QPixmap, QImageWriter
from PyQt6.QtCore import QSettings
from diffusers.models import AutoencoderKL

from iartisanxl.modules.base_module import BaseModule
from iartisanxl.modules.common.drop_lightbox import DropLightBox
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.prompt_window import PromptWindow
from iartisanxl.modules.common.panels.generation_panel import GenerationPanel
from iartisanxl.modules.common.panels.lora_panel import LoraPanel
from iartisanxl.modules.common.diffusers_utils import load_vae_from_safetensors
from iartisanxl.menu.right_menu import RightMenu
from iartisanxl.generation.generation_data_object import ImageGenData
from iartisanxl.generation.model_data_object import ModelDataObject
from iartisanxl.generation.vae_data_object import VaeDataObject
from iartisanxl.generation.schedulers.schedulers_utils import load_scheduler
from iartisanxl.generation.schedulers.schedulers import schedulers
from iartisanxl.console.console_stream import ConsoleStream
from iartisanxl.threads.pipeline_setup_thread import PipelineSetupThread
from iartisanxl.threads.lora_setup_thread import LoraSetupThread
from iartisanxl.threads.image_generation_thread import ImageGenerationThread
from iartisanxl.threads.taesd_loader_thread import TaesdLoaderThread
from iartisanxl.pipelines.txt_pipeline import ImageArtisanTextPipeline


class TextToImageModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger()

        self.settings = QSettings("ZCode", "ImageArtisanXL")

        self.settings.beginGroup("text_to_image")
        self.module_options = {
            "right_menu_expanded": self.settings.value(
                "right_menu_expanded", True, type=bool
            ),
            "positive_prompt_split": self.settings.value(
                "positive_prompt_split", False, type=bool
            ),
            "negative_prompt_split": self.settings.value(
                "negative_prompt_split", False, type=bool
            ),
        }
        model_name = self.settings.value("model_name", "no model selected", type=str)
        self.settings.endGroup()

        self.settings.beginGroup("image_generation")
        model_name = self.settings.value("model_name", "no model selected", type=str)
        model_path = self.settings.value("model_path", "", type=str)
        model_type = self.settings.value("model_type", "", type=str)
        model_version = self.settings.value("model_version", "", type=str)
        model = ModelDataObject(
            name=model_name, path=model_path, type=model_type, version=model_version
        )

        vae_name = self.settings.value("vae_name", "Model default", type=str)
        vae_path = self.settings.value("vae_path", "", type=str)
        vae = VaeDataObject(name=vae_name, path=vae_path)

        self.setAcceptDrops(True)

        self.image_generation_data = ImageGenData(
            module="texttoimage",
            seed=0,
            image_width=self.settings.value("image_width", 1024, type=int),
            image_height=self.settings.value("image_height", 1024, type=int),
            steps=self.settings.value("steps", 20, type=int),
            guidance=self.settings.value("guidance", 7.5, type=float),
            base_scheduler=self.settings.value("base_scheduler", 0, type=int),
            lora_scale=1.0,
            loras=[],
            model=model,
            vae=vae,
            positive_prompt_clipl="",
            positive_prompt_clipg="",
            negative_prompt_clipl="",
            negative_prompt_clipg="",
            clip_skip=self.settings.value("clip_skip", 0, type=int),
        )
        self.settings.endGroup()

        self.rendering_generation_data = None
        self.observers = []

        self.base_pipeline = None
        self.new_pipeline = False
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # pylint: disable=no-member
        self.batch_size = 1
        self.taesd_dec = None
        self.changed_parameters = []

        self.taesd_loader_thread = None
        self.pipeline_setup_thread = None
        self.lora_setup_thread = None
        self.image_generation_thread = None

        self.threads = {
            "taesd_loader_thread": self.taesd_loader_thread,
            "pipeline_setup_thread": self.pipeline_setup_thread,
            "lora_setup_thread": self.lora_setup_thread,
            "image_generation_thread": self.image_generation_thread,
        }

        self.generating = False
        self.auto_save = False
        self.continuous_generation = False

        self.init_ui()

        self.console_stream = ConsoleStream()
        # sys.stdout = self.console_stream
        # sys.stderr = self.console_stream

    def init_ui(self):
        super().init_ui()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        self.image_viewer = ImageViewerSimple(self.directories.outputs_images)
        top_layout.addWidget(self.image_viewer)
        main_layout.addLayout(top_layout)

        spacer = QSpacerItem(
            5,
            5,
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        main_layout.addSpacerItem(spacer)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.prompt_window = PromptWindow(
            self.image_generation_data, self.module_options
        )
        self.prompt_window.generate_signal.connect(self.generation_clicked)
        main_layout.addWidget(self.prompt_window)
        self.subscribe(self.prompt_window)

        right_menu = RightMenu(
            self.module_options,
            self.directories,
            self.image_generation_data,
            self.image_viewer,
            self.prompt_window,
            self.auto_generate,
            self.open_dialog,
        )
        top_layout.addWidget(right_menu)
        self.subscribe(right_menu)
        top_layout.setStretch(0, 1)

        # Add the panels to the menu
        right_menu.add_panel(
            "Generation", GenerationPanel, schedulers, self.module_options
        )
        right_menu.add_panel("LoRAs", LoraPanel)

        main_layout.setStretch(0, 16)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 4)
        self.setLayout(main_layout)

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop image here")

    def closeEvent(self, event):
        self.settings.beginGroup("text_to_image")
        self.settings.setValue(
            "right_menu_expanded", self.module_options.get("right_menu_expanded")
        )
        self.settings.setValue(
            "positive_prompt_split",
            self.module_options.get("positive_prompt_split"),
        )
        self.settings.setValue(
            "negative_prompt_split",
            self.module_options.get("negative_prompt_split"),
        )
        self.settings.endGroup()

        self.settings.beginGroup("image_generation")
        if self.image_generation_data.model is not None:
            self.settings.setValue("model_name", self.image_generation_data.model.name)
            self.settings.setValue("model_path", self.image_generation_data.model.path)
            self.settings.setValue("model_type", self.image_generation_data.model.type)
            self.settings.setValue(
                "model_version", self.image_generation_data.model.version
            )
        else:
            self.settings.setValue("model_name", "no model selected")
            self.settings.setValue("model_path", "")
            self.settings.setValue("model_type", "")
            self.settings.setValue("model_version", "")

        self.settings.setValue("vae_name", self.image_generation_data.vae.name)
        self.settings.setValue("vae_path", self.image_generation_data.vae.path)
        self.settings.setValue(
            "base_scheduler", self.image_generation_data.base_scheduler
        )
        self.settings.setValue("image_width", self.image_generation_data.image_width)
        self.settings.setValue("image_height", self.image_generation_data.image_height)
        self.settings.setValue("guidance", self.image_generation_data.guidance)
        self.settings.setValue("steps", self.image_generation_data.steps)
        self.settings.setValue("clip_skip", self.image_generation_data.clip_skip)
        self.settings.endGroup()

        self.base_pipeline = None
        self.device = None
        self.taesd_dec = None

        self.taesd_loader_thread = None
        self.pipeline_setup_thread = None
        self.lora_setup_thread = None
        self.image_generation_thread = None

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        super().closeEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_lightbox.show()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_lightbox.hide()
        event.accept()

    def dropEvent(self, event):
        self.drop_lightbox.hide()

        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.endswith(".png"):
                try:
                    image = Image.open(path)
                    metadata = image.info
                    serialized_data = metadata.get("data")

                    self.update_status_bar("Getting generation data from image...")

                    if serialized_data is None:
                        self.update_status_bar("No metadata found in the image.")

                    self.logger.debug(serialized_data)

                    self.update_status_bar(
                        "Setting up generation from metada found in the image..."
                    )
                    self.image_generation_data.loras = []
                    error = self.image_generation_data.update_attributes(
                        self.deserialize_image_data(serialized_data)
                    )
                    if error is not None:
                        self.show_error(error)
                    self.update_status_bar("Ready")
                    self.notify_observers()
                    self.prompt_window.unblock_seed()
                except json.JSONDecodeError as json_error:
                    self.logger.error(
                        "Error decoding JSON from image metadata with this error: %s",
                        json_error,
                    )
                    self.logger.debug("JSONDecodeError exception", exc_info=True)
                except ValueError as data_error:
                    self.logger.error(
                        "Value error from image metadata with this message: %s",
                        data_error,
                    )
                    self.logger.debug("ValueError exception", exc_info=True)

    def generation_clicked(
        self, auto_save: bool = False, continuous_generation: bool = False
    ):
        self.auto_save = auto_save
        self.continuous_generation = continuous_generation

        if self.generating:
            self.on_abort()
            return

        if self.image_generation_data.model.name == "no model selected":
            self.show_snackbar("You need to first choose a Stable Diffusion XL model.")
            return

        if len(self.image_generation_data.positive_prompt_clipg) == 0:
            self.show_snackbar("You forgot to write a prompt.")
            return

        if self.image_generation_data.model is None:
            self.show_snackbar("No base model selected.")
            return

        self.generating = True
        self.prompt_window.set_button_abort()

        if (
            self.image_generation_data.seed <= 0
            or self.prompt_window.random_checkbox.isChecked()
        ):
            self.image_generation_data.seed = random.randint(0, 2**32 - 1)
            self.prompt_window.seed_text.setText(str(self.image_generation_data.seed))

        self.progress_bar.setValue(0)

        if self.preferences.intermediate_images:
            if self.taesd_dec is None:
                self.status_bar.showMessage("Loading taesd...")
                self.taesd_loader_thread = TaesdLoaderThread(self.device)
                self.threads["taesd_loader_thread"] = self.taesd_loader_thread
                self.taesd_loader_thread.taesd_loaded.connect(self.taesd_loaded)
                self.taesd_loader_thread.taesd_error.connect(self.taesd_error)
                self.taesd_loader_thread.finished.connect(self.reset_thread)
                self.taesd_loader_thread.start()
                return
        else:
            if self.taesd_dec is not None:
                self.taesd_dec = None
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        self.setup_pipeline()

    def taesd_error(self, error_text):
        self.show_error(error_text)
        self.preferences.intermediate_images = False
        self.setup_pipeline()

    def taesd_loaded(self):
        self.taesd_dec = self.taesd_loader_thread.taesd_dec
        self.status_bar.showMessage("Taesd loaded")
        self.setup_pipeline()

    def setup_pipeline(self):
        if self.rendering_generation_data is not None:
            self.changed_parameters = []

            for attr in ImageGenData.__slots__:
                if attr == "_loras":
                    # Check if the number of Lora objects has changed
                    if len(self.image_generation_data.loras) != len(
                        self.rendering_generation_data.loras
                    ):
                        self.changed_parameters.append(attr)
                    else:
                        # Check if the attributes of each Lora object have changed
                        for i, lora in enumerate(self.image_generation_data.loras):
                            for lora_attr in lora.__dict__:
                                if getattr(lora, lora_attr) != getattr(
                                    self.rendering_generation_data.loras[i], lora_attr
                                ):
                                    self.changed_parameters.append(attr)
                                    break
                else:
                    if getattr(self.image_generation_data, attr) != getattr(
                        self.rendering_generation_data, attr
                    ):
                        self.changed_parameters.append(attr)

        self.rendering_generation_data = self.image_generation_data.copy()
        self.logger.debug(
            "The following parameters have changed: %s", self.changed_parameters
        )

        use_model_offload = False
        use_sequential_offload = False

        if self.base_pipeline is not None:
            if self.preferences.sequential_offload:
                if not self.base_pipeline.sequential_cpu_offloaded:
                    use_sequential_offload = True
                    self.base_pipeline = None
            else:
                if self.base_pipeline.sequential_cpu_offloaded:
                    self.base_pipeline = None
                    self.logger.debug(
                        "Sequential cpu offload disabled, reloading model."
                    )

                    if self.preferences.model_offload:
                        use_model_offload = True
                else:
                    if self.preferences.model_offload:
                        if not self.base_pipeline.model_cpu_offloaded:
                            use_model_offload = True
                            self.base_pipeline = None
                    else:
                        if self.base_pipeline.model_cpu_offloaded:
                            self.base_pipeline = None
                            self.logger.debug(
                                "Model cpu offload disabled, reloading model."
                            )

        if self.base_pipeline is None or "_model" in self.changed_parameters:
            self.status_bar.showMessage("Setting up the pipeline...")
            self.base_pipeline = None
            self.lora_setup_thread = None
            self.image_generation_thread = None
            self.new_pipeline = True
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            if self.preferences.sequential_offload:
                use_sequential_offload = True
            else:
                if self.preferences.model_offload:
                    use_model_offload = True

            self.pipeline_setup_thread = PipelineSetupThread(
                self.rendering_generation_data,
                model_offload=use_model_offload,
                sequential_offload=use_sequential_offload,
            )
            self.threads["pipeline_setup_thread"] = self.pipeline_setup_thread
            self.pipeline_setup_thread.pipeline_ready.connect(self.pipeline_ready)
            self.pipeline_setup_thread.pipeline_error.connect(self.show_error)
            self.pipeline_setup_thread.status_changed.connect(self.update_status_bar)
            self.pipeline_setup_thread.finished.connect(self.reset_thread)
            self.pipeline_setup_thread.start()
        else:
            self.new_pipeline = False
            self.pipeline_ready(self.base_pipeline)

    # pylint: disable=no-member
    def pipeline_ready(self, pipeline: Optional[ImageArtisanTextPipeline] = None):
        if pipeline is not None:
            self.base_pipeline = pipeline

            if "_vae" in self.changed_parameters:
                self.update_status_bar("Changing to selected vae...")
                try:
                    if len(self.rendering_generation_data.vae.path) > 0:
                        vae = AutoencoderKL.from_pretrained(
                            self.rendering_generation_data.vae.path,
                            torch_dtype=torch.float16,
                        )
                    else:
                        if self.rendering_generation_data.model.type == "diffusers":
                            self.update_status_bar("Changing to model vae...")
                            vae = AutoencoderKL.from_pretrained(
                                self.rendering_generation_data.model.path,
                                torch_dtype=torch.float16,
                                subfolder="vae",
                                variant="fp16",
                                use_safetensors=True,
                            )
                        else:
                            self.update_status_bar("Changing to model vae...")
                            vae = load_vae_from_safetensors(
                                self.rendering_generation_data.model.path,
                                "./configs/sd_xl_base.yaml",
                            )

                    self.base_pipeline.vae = vae
                except AttributeError as attribute_error:
                    self.show_error(f"{attribute_error}", True)
                    self.logger.error(
                        "Attribute error trying to load the vae: %s",
                        attribute_error,
                    )
                    self.logger.debug("AttributeError exception", exc_info=True)
                    return
                except OSError as os_error:
                    self.show_error(f"{os_error}", True)
                    self.show_error(f"{os_error}", True)
                    self.logger.error(
                        "OS error trying to load the vae: %s",
                        os_error,
                    )
                    self.logger.debug("OSError exception", exc_info=True)
                    return

            if "_base_scheduler" in self.changed_parameters:
                self.update_status_bar("Changing the scheduler...")
                self.base_pipeline.scheduler = load_scheduler(
                    self.rendering_generation_data.base_scheduler
                )

            if self.preferences.use_tomes:
                self.status_bar.showMessage("Token merging is active...")
                tomesd.apply_patch(self.base_pipeline, ratio=0.5)
            else:
                self.status_bar.showMessage("Token merging is disabled...")
                tomesd.remove_patch(self.base_pipeline)

            if self.new_pipeline or "_loras" in self.changed_parameters:
                self.lora_setup_thread = LoraSetupThread(
                    self.base_pipeline, self.rendering_generation_data.loras
                )
                self.threads["lora_setup_thread"] = self.lora_setup_thread
                self.lora_setup_thread.status_changed.connect(self.update_status_bar)
                self.lora_setup_thread.loras_error.connect(self.show_error)
                self.lora_setup_thread.loras_ready.connect(self.start_generation)
                self.lora_setup_thread.finished.connect(self.reset_thread)
                self.lora_setup_thread.lora_setup_aborted.connect(
                    self.on_finished_abort
                )
                self.lora_setup_thread.start()
            else:
                self.start_generation()

    def start_generation(self):
        self.status_bar.showMessage("Setting up generation...")
        self.progress_bar.setMaximum(self.rendering_generation_data.steps)

        self.image_generation_thread = ImageGenerationThread(
            self.base_pipeline, self.rendering_generation_data
        )
        self.threads["image_generation_thread"] = self.image_generation_thread
        self.image_generation_thread.progress_update.connect(self.step_progress_update)
        self.image_generation_thread.status_changed.connect(self.update_status_bar)
        self.image_generation_thread.generation_error.connect(self.show_error)
        self.image_generation_thread.generation_aborted.connect(self.on_finished_abort)
        self.image_generation_thread.generation_finished.connect(
            self.generation_finished
        )
        self.image_generation_thread.finished.connect(self.reset_thread)
        self.image_generation_thread.start()

    def step_progress_update(self, step, latents):
        self.progress_bar.setValue(step)

        if self.preferences.intermediate_images and self.taesd_dec is not None:
            with torch.no_grad():
                decoded = (
                    self.taesd_dec(latents.float()).clamp(0, 1).mul_(255).round().byte()
                )
            image = Image.fromarray(decoded[0].permute(1, 2, 0).cpu().numpy())

            # Convert the image to a QImage
            qimage = QImage(
                image.tobytes("raw", "RGB"),
                image.width,
                image.height,
                QImage.Format.Format_RGB888,
            )

            # Convert the QImage to a QPixmap
            qpixmap = QPixmap.fromImage(qimage)
            self.image_viewer.set_pixmap(qpixmap)

    def serialize_image_data(self, rendering_generation_data: ImageGenData) -> str:
        data = {
            attr.strip("_"): getattr(rendering_generation_data, attr)
            for attr in rendering_generation_data.__slots__
            if attr not in ("_loras", "_model")
        }
        data["loras"] = [
            {
                "name": lora.name,
                "filename": lora.filename,
                "version": lora.version,
                "path": lora.path,
                "weight": lora.weight,
            }
            for lora in rendering_generation_data.loras
        ]
        data["model"] = {
            "name": rendering_generation_data.model.name,
            "path": rendering_generation_data.model.path,
            "type": rendering_generation_data.model.type,
            "version": rendering_generation_data.model.version,
        }
        data["vae"] = {
            "name": rendering_generation_data.vae.name,
            "path": rendering_generation_data.vae.path,
        }
        serialized_data = json.dumps(data)
        return serialized_data

    def deserialize_image_data(self, serialized_data: str) -> ImageGenData:
        data = json.loads(serialized_data)
        data = {key.strip("_"): value for key, value in data.items()}
        return data

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)

    def generation_finished(self, image, duration: float = None):
        if duration is not None:
            self.status_bar.showMessage(
                f"Ready - {round(duration, 1)} s ({round(duration * 1000, 2)} ms)"
            )
        else:
            self.status_bar.showMessage("Ready")

        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)

        image_data = image.tobytes()
        image = None

        qimage = QImage(
            image_data,
            self.rendering_generation_data.image_width,
            self.rendering_generation_data.image_height,
            QImage.Format.Format_RGB888,
        )

        qpixmap = QPixmap.fromImage(qimage)
        self.image_viewer.set_pixmap(qpixmap)
        self.image_viewer.serialized_data = self.serialize_image_data(
            self.rendering_generation_data
        )

        if self.auto_save:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            writer = QImageWriter(
                os.path.join(self.directories.outputs_images, f"{timestamp}.png"),
                b"png",
            )
            writer.setText("data", self.image_viewer.serialized_data)
            writer.write(qimage)

        self.prompt_window.set_button_generate()
        self.generating = False

        if self.continuous_generation:
            self.generation_clicked(self.auto_save, self.continuous_generation)

    def subscribe(self, observer):
        self.observers.append(observer)

    def unsubscribe(self, observer):
        self.observers.remove(observer)

    def notify_observers(self):
        self.logger.debug("Notifying observers: %s", self.observers)
        for observer in self.observers:
            observer.update_ui()

    def update_status_bar(self, text):
        self.status_bar.showMessage(text)

    def show_error(self, text, empty_pipeline: bool = False):
        if empty_pipeline:
            self.base_pipeline = None
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self.show_snackbar(text)
        self.status_bar.showMessage(f"Error: {text}")
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.prompt_window.set_button_generate()
        self.generating = False

    def reset_thread(self):
        sender = self.sender()
        for name, thread in self.threads.items():
            if thread == sender:
                self.threads[name] = None
                break

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def auto_generate(self, generation_data):
        self.image_generation_data.loras = []
        error = self.image_generation_data.update_attributes(
            self.deserialize_image_data(generation_data)
        )
        if error is not None:
            self.show_error(error)

        self.notify_observers()
        self.prompt_window.unblock_seed()
        self.generation_clicked()

    def on_abort(self):
        if self.pipeline_setup_thread is not None:
            self.pipeline_setup_thread.abort = True

        if self.lora_setup_thread is not None:
            self.lora_setup_thread.abort = True

        if self.image_generation_thread is not None:
            self.image_generation_thread.abort_generation()

    def on_finished_abort(self):
        self.update_status_bar("Aborted")
        self.prompt_window.set_button_generate()
        self.generating = False
