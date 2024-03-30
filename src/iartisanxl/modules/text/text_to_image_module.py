import logging
import random

import torch
from PIL import Image
from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QImage, QPixmap

# import tomesd
from PyQt6.QtWidgets import QHBoxLayout, QProgressBar, QSizePolicy, QSpacerItem, QVBoxLayout

from iartisanxl.console.console_stream import ConsoleStream
from iartisanxl.generation.adapter_list import AdapterList
from iartisanxl.generation.image_generation_data import ImageGenerationData
from iartisanxl.generation.model_data_object import ModelDataObject
from iartisanxl.generation.schedulers.schedulers import schedulers
from iartisanxl.generation.vae_data_object import VaeDataObject
from iartisanxl.menu.right_menu import RightMenu
from iartisanxl.modules.base_module import BaseModule
from iartisanxl.modules.common.controlnet.controlnet_data import ControlNetData
from iartisanxl.modules.common.controlnet.controlnet_panel import ControlNetPanel
from iartisanxl.modules.common.drop_lightbox import DropLightBox
from iartisanxl.modules.common.image.image_processor import ImageProcessor
from iartisanxl.modules.common.image_viewer_simple import ImageViewerSimple
from iartisanxl.modules.common.ip_adapter.ip_adapter_data_object import IPAdapterDataObject
from iartisanxl.modules.common.ip_adapter.ip_adapter_panel import IPAdapterPanel
from iartisanxl.modules.common.lora.lora_data_object import LoraDataObject
from iartisanxl.modules.common.lora.lora_list import LoraList
from iartisanxl.modules.common.lora.lora_panel import LoraPanel
from iartisanxl.modules.common.panels.generation_panel import GenerationPanel
from iartisanxl.modules.common.prompt_window import PromptWindow
from iartisanxl.modules.common.t2i_adapter.t2i_adapter_data_object import T2IAdapterDataObject
from iartisanxl.modules.common.t2i_adapter.t2i_adapter_panel import T2IAdapterPanel
from iartisanxl.threads.image_processor_thread import ImageProcesorThread
from iartisanxl.threads.node_graph_thread import NodeGraphThread
from iartisanxl.threads.taesd_loader_thread import TaesdLoaderThread


class TextToImageModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger()

        self.settings = QSettings("ZCode", "ImageArtisanXL")

        self.settings.beginGroup("text_to_image")
        self.module_options = {
            "right_menu_expanded": self.settings.value("right_menu_expanded", True, type=bool),
            "positive_prompt_split": self.settings.value("positive_prompt_split", False, type=bool),
            "negative_prompt_split": self.settings.value("negative_prompt_split", False, type=bool),
        }
        model_name = self.settings.value("model_name", "no model selected", type=str)
        self.settings.endGroup()

        self.settings.beginGroup("image_generation")
        model_name = self.settings.value("model_name", "no model selected", type=str)
        model_path = self.settings.value("model_path", "", type=str)
        model_type = self.settings.value("model_type", "", type=str)
        model_version = self.settings.value("model_version", "", type=str)
        model = ModelDataObject(name=model_name, path=model_path, type=model_type, version=model_version)
        vae_name = self.settings.value("vae_name", "Model default", type=str)
        vae_path = self.settings.value("vae_path", "", type=str)
        vae = VaeDataObject(name=vae_name, path=vae_path)

        self.setAcceptDrops(True)

        self.lora_list = LoraList()
        self.controlnet_list = AdapterList[ControlNetData]()
        self.t2i_adapter_list = AdapterList[T2IAdapterDataObject]()
        self.ip_adapter_list = AdapterList[IPAdapterDataObject]()
        self.image_generation_data = ImageGenerationData(
            module="texttoimage",
            seed=0,
            image_width=self.settings.value("image_width", 1024, type=int),
            image_height=self.settings.value("image_height", 1024, type=int),
            steps=self.settings.value("steps", 20, type=int),
            guidance=self.settings.value("guidance", 7.5, type=float),
            base_scheduler=self.settings.value("base_scheduler", 0, type=int),
            lora_scale=1.0,
            model=model,
            vae=vae,
            positive_prompt_clipl="",
            positive_prompt_clipg="",
            negative_prompt_clipl="",
            negative_prompt_clipg="",
            clip_skip=self.settings.value("clip_skip", 0, type=int),
        )
        self.image_generation_data.update_previous_state()
        self.settings.endGroup()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.bfloat16
        self.batch_size = 1
        self.taesd_dec = None

        self.generating = False
        self.auto_save = False
        self.continuous_generation = False

        self.event_bus.subscribe("image_generation_data", self.on_image_generation_data)
        self.event_bus.subscribe("auto_generate", self.on_auto_generate)

        self.init_ui()

        self.taesd_loader_thread = None
        self.image_processor_thread = None
        self.node_graph_thread = NodeGraphThread(node_graph=self.node_graph, torch_dtype=self.torch_dtype)
        self.node_graph_thread.progress_update.connect(self.step_progress_update)
        self.node_graph_thread.status_changed.connect(self.update_status_bar)
        self.node_graph_thread.generation_error.connect(self.show_error)
        self.node_graph_thread.generation_aborted.connect(self.on_finished_abort)
        self.node_graph_thread.generation_finished.connect(self.generation_finished)

        self.threads = {
            "taesd_loader_thread": self.taesd_loader_thread,
            "image_processor_thread": self.image_processor_thread,
        }

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
        self.image_viewer = ImageViewerSimple(self.directories.outputs_images, self.preferences)
        top_layout.addWidget(self.image_viewer)
        main_layout.addLayout(top_layout)

        spacer = QSpacerItem(5, 5, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        main_layout.addSpacerItem(spacer)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.prompt_window = PromptWindow(self.image_generation_data, self.module_options)
        self.prompt_window.generate_signal.connect(self.generation_clicked)
        main_layout.addWidget(self.prompt_window)

        self.right_menu = RightMenu(
            self.module_options,
            self.preferences,
            self.directories,
            self.image_generation_data,
            self.lora_list,
            self.controlnet_list,
            self.t2i_adapter_list,
            self.ip_adapter_list,
            self.image_viewer,
            self.prompt_window,
            self.show_error,
        )
        top_layout.addWidget(self.right_menu)
        top_layout.setStretch(0, 1)

        # Add the panels to the menu
        self.right_menu.add_panel("Generation", GenerationPanel, schedulers)
        self.right_menu.add_panel("LoRAs", LoraPanel)
        self.right_menu.add_panel("ControlNet", ControlNetPanel)
        self.right_menu.add_panel("T2I Adapters", T2IAdapterPanel)
        self.right_menu.add_panel("IP Adapters", IPAdapterPanel)

        main_layout.setStretch(0, 16)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 4)
        self.setLayout(main_layout)

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop image here")

    def closeEvent(self, event):
        self.settings.beginGroup("text_to_image")
        self.settings.setValue("right_menu_expanded", self.module_options.get("right_menu_expanded"))
        self.settings.setValue("positive_prompt_split", self.module_options.get("positive_prompt_split"))
        self.settings.setValue("negative_prompt_split", self.module_options.get("negative_prompt_split"))
        self.settings.endGroup()

        self.settings.beginGroup("image_generation")
        if self.image_generation_data.model is not None:
            self.settings.setValue("model_name", self.image_generation_data.model.name)
            self.settings.setValue("model_path", self.image_generation_data.model.path)
            self.settings.setValue("model_type", self.image_generation_data.model.type)
            self.settings.setValue("model_version", self.image_generation_data.model.version)
        else:
            self.settings.setValue("model_name", "no model selected")
            self.settings.setValue("model_path", "")
            self.settings.setValue("model_type", "")
            self.settings.setValue("model_version", "")

        self.settings.setValue("vae_name", self.image_generation_data.vae.name)
        self.settings.setValue("vae_path", self.image_generation_data.vae.path)
        self.settings.setValue("base_scheduler", self.image_generation_data.base_scheduler)
        self.settings.setValue("image_width", self.image_generation_data.image_width)
        self.settings.setValue("image_height", self.image_generation_data.image_height)
        self.settings.setValue("guidance", self.image_generation_data.guidance)
        self.settings.setValue("steps", self.image_generation_data.steps)
        self.settings.setValue("clip_skip", self.image_generation_data.clip_skip)
        self.settings.endGroup()

        self.right_menu.close_all_dialogs()
        self.node_graph_thread.clean_up()

        self.image_generation_data = None
        self.node_graph = None
        self.device = None
        self.torch_dtype = None
        self.taesd_dec = None
        self.lora_list = None
        self.controlnet_list = None
        self.t2i_adapter_list = None

        self.taesd_loader_thread = None
        self.node_graph_thread = None
        self.image_processor_thread = None

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
                self.image_processor_thread = ImageProcesorThread(path)
                self.threads["image_processor_thread"] = self.image_processor_thread
                self.image_processor_thread.serialized_data_obtained.connect(self.on_serialized_data_obtained)
                self.image_processor_thread.status_changed.connect(self.update_status_bar)
                self.image_processor_thread.image_loaded.connect(self.on_dropped_image_loaded)
                self.image_processor_thread.image_error.connect(self.show_error)
                self.image_processor_thread.finished.connect(self.reset_thread)
                self.image_processor_thread.start()

    def on_image_generation_data(self, data):
        if hasattr(self.image_generation_data, data["attr"]):
            setattr(self.image_generation_data, data["attr"], data["value"])

    def on_serialized_data_obtained(self, json_graph):
        loras = self.image_generation_data.update_from_json(json_graph)

        if len(loras) > 0:
            self.lora_list.clear_loras()

            for lora in loras:
                lora_object = LoraDataObject(
                    name=lora["lora_name"],
                    filename=lora["name"],
                    version=lora["version"],
                    path=lora["path"],
                )
                lora_object.set_weights(lora["scale"])
                self.lora_list.add(lora_object)

        self.event_bus.publish("update_from_json", {})
        self.lora_list.dropped_image = True
        self.prompt_window.unblock_seed()

    def on_dropped_image_loaded(self, image: QPixmap):
        self.image_viewer.set_pixmap(image)
        self.update_status_bar("Ready")

    def generation_clicked(self, auto_save: bool = False, continuous_generation: bool = False):
        self.auto_save = auto_save
        self.continuous_generation = continuous_generation

        if self.generating:
            self.on_abort()
            return

        if self.image_generation_data.model.name == "no model selected":
            self.show_snackbar("You need to first choose a Stable Diffusion XL model.")
            return

        if self.image_generation_data.model is None:
            self.show_snackbar("No base model selected.")
            return

        self.generating = True
        self.prompt_window.set_button_abort()

        if self.image_generation_data.seed <= 0 or self.prompt_window.random_checkbox.isChecked():
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

        self.run_graph()

    def taesd_error(self, error_text):
        self.show_error(error_text)
        self.preferences.intermediate_images = False
        self.run_graph()

    def taesd_loaded(self):
        self.taesd_dec = self.taesd_loader_thread.taesd_dec
        self.status_bar.showMessage("Taesd loaded")
        self.run_graph()

    def run_graph(self):
        self.status_bar.showMessage("Setting up generation...")
        self.progress_bar.setMaximum(self.image_generation_data.steps)

        self.node_graph_thread.directories = self.directories
        self.node_graph_thread.image_generation_data = self.image_generation_data
        self.node_graph_thread.lora_list = self.lora_list
        self.node_graph_thread.controlnet_list = self.controlnet_list
        self.node_graph_thread.t2i_adapter_list = self.t2i_adapter_list
        self.node_graph_thread.ip_adapter_list = self.ip_adapter_list
        self.node_graph_thread.model_offload = self.preferences.model_offload
        self.node_graph_thread.sequential_offload = self.preferences.sequential_offload

        self.node_graph_thread.start()

    def step_progress_update(self, step, latents):
        self.progress_bar.setValue(step)

        if self.preferences.intermediate_images and self.taesd_dec is not None:
            with torch.no_grad():
                decoded = self.taesd_dec(latents.float()).clamp(0, 1).mul_(255).round().byte()
            image = Image.fromarray(decoded[0].permute(1, 2, 0).cpu().numpy())

            # Convert the image to a QImage
            qimage = QImage(image.tobytes("raw", "RGB"), image.width, image.height, QImage.Format.Format_RGB888)

            # Convert the QImage to a QPixmap
            qpixmap = QPixmap.fromImage(qimage)
            self.image_viewer.set_pixmap(qpixmap)

    def generation_finished(self, image: Image):
        image_generation_node = self.node_graph.get_node_by_name("image_generation")
        duration = image_generation_node.elapsed_time

        if duration is not None:
            self.status_bar.showMessage(f"Ready - {round(duration, 1)} s ({round(duration * 1000, 2)} ms)")
        else:
            self.status_bar.showMessage("Ready")

        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)

        image_processor = ImageProcessor()
        image_processor.set_pillow_image(image)
        image = None

        self.image_viewer.set_pixmap(image_processor.get_qpixmap())
        serialized_data = self.node_graph.to_json()
        self.image_viewer.serialized_data = serialized_data

        if self.auto_save:
            if self.preferences.save_image_metadata:
                image_processor.set_serialized_data(serialized_data)
            image_processor.save_to_png(self.directories.outputs_images)

        self.prompt_window.set_button_generate()
        self.generating = False

        if self.continuous_generation:
            self.generation_clicked(self.auto_save, self.continuous_generation)

    def show_error(self, text):
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

    def on_auto_generate(self, data):
        generation_data = data.get("generation_data")
        image = ImageProcessor()
        image.serialized_data = generation_data
        loras = self.image_generation_data.update_from_json(generation_data)

        if len(loras) > 0:
            self.lora_list.clear_loras()

            for lora in loras:
                lora_object = LoraDataObject(
                    name=lora["lora_name"],
                    filename=lora["name"],
                    version=lora["version"],
                    path=lora["path"],
                )
                lora_object.set_weights(lora["scale"])
                self.lora_list.add(lora_object)

        self.event_bus.publish("update_from_json", {})
        self.lora_list.dropped_image = True
        self.prompt_window.unblock_seed()
        self.generation_clicked()

    def on_abort(self):
        self.node_graph_thread.abort_graph()

    def on_finished_abort(self):
        self.update_status_bar("Aborted")
        self.prompt_window.set_button_generate()
        self.generating = False
