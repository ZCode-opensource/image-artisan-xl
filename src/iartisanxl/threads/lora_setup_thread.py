import logging

from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.pipelines.txt_pipeline import ImageArtisanTextPipeline
from iartisanxl.generation.lora_data_object import LoraDataObject


class LoraSetupThread(QThread):
    loras_ready = pyqtSignal()
    status_changed = pyqtSignal(str)
    loras_error = pyqtSignal(str)
    lora_setup_aborted = pyqtSignal()

    def __init__(
        self,
        pipeline: ImageArtisanTextPipeline,
        loras: list[LoraDataObject],
        deleted_loras: list[str],
    ):
        super().__init__()

        self.logger = logging.getLogger()

        self.pipeline = pipeline
        self.loras = loras
        self.deleted_loras = deleted_loras
        self.abort = False

    def run(self):
        self.status_changed.emit("Loading LoRAs...")

        if len(self.loras) > 0:
            self.logger.debug("Deleting adapters: %s", self.deleted_loras)
            self.pipeline.delete_adapters(self.deleted_loras)

            names = []
            weights = []

            for lora in self.loras:
                if lora.enabled:
                    lora_name = lora.filename.replace(".", "_")
                    if lora_name not in getattr(self.pipeline.unet, "peft_config", {}):
                        try:
                            if self.abort:
                                self.lora_setup_aborted.emit()
                                return
                            self.status_changed.emit(f"Loading LoRA: {lora_name}")
                            self.pipeline.load_lora_weights(
                                lora.path, adapter_name=lora_name
                            )
                            if self.abort:
                                self.lora_setup_aborted.emit()
                                return
                        except TypeError as e:
                            self.loras_error.emit(
                                f"Runtime error: {lora_name} is an incompatible LoRA."
                            )
                            self.logger.error(
                                "Type error when trying to load the LoRA %s with this message: %s",
                                lora_name,
                                e,
                            )
                            self.logger.debug("Exception occurred", exc_info=True)
                            return
                        except RuntimeError as e:
                            self.loras_error.emit(
                                f"Runtime error: {lora_name} is an incompatible LoRA."
                            )
                            self.logger.error(
                                "Runtime error when trying to load the LoRA %s with this message: %s",
                                lora_name,
                                e,
                            )
                            self.logger.debug("Exception occurred", exc_info=True)
                            return
                        except ValueError as e:
                            self.loras_error.emit(
                                "LoRA not loaded, try to remove it and add it again."
                            )
                            self.logger.error(
                                "Value error when trying to load the LoRA %s with this message: %s",
                                lora_name,
                                e,
                            )
                            self.logger.debug("Exception occurred", exc_info=True)
                            return
                        except OSError as e:
                            self.loras_error.emit(f"OSError: {lora_name} not found.")
                            self.logger.error(
                                "OS error when trying to load the LoRA %s with this message: %s",
                                lora_name,
                                e,
                            )
                            self.logger.debug("Exception occurred", exc_info=True)
                            return

                    names.append(lora_name)
                    weights.append(lora.weight)

            self.logger.debug("%s %s", names, weights)

            try:
                self.pipeline.set_adapters(names, adapter_weights=weights)
            except KeyError as e:
                self.loras_error.emit(f"The key {e} does not exist.")
                self.logger.error("Exception occurred", exc_info=True)
                return
        else:
            if self.abort:
                self.lora_setup_aborted.emit()
                return
            self.pipeline.unload_lora_weights()

        self.loras_ready.emit()
