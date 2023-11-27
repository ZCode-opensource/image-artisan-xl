import logging

import torch
from PIL import Image

from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.generation.image_generation_data import ImageGenerationData
from iartisanxl.generation.lora_list import LoraList
from iartisanxl.graph.iartisanxl_node_graph import ImageArtisanNodeGraph
from iartisanxl.graph.nodes.lora_node import LoraNode


class NodeGraphThread(QThread):
    status_changed = pyqtSignal(str)
    progress_update = pyqtSignal(int, torch.Tensor)
    generation_finished = pyqtSignal(Image.Image)
    generation_error = pyqtSignal(str, bool)
    generation_aborted = pyqtSignal()

    def __init__(
        self,
        node_graph: ImageArtisanNodeGraph = None,
        image_generation_data: ImageGenerationData = None,
        lora_list: LoraList = None,
        model_offload: bool = False,
        sequential_offload: bool = False,
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.logger = logging.getLogger()
        self.node_graph = node_graph
        self.image_generation_data = image_generation_data
        self.lora_list = lora_list
        self.model_offload = model_offload
        self.sequential_offload = sequential_offload
        self.torch_dtype = torch_dtype
        self.abort = False

    def run(self):
        self.status_changed.emit("Generating image...")

        if self.node_graph is None:
            self.node_graph = self.image_generation_data.create_text_to_image_graph()

            # connect the essential callbacks
            image_generation = self.node_graph.get_node_by_name("image_generation")
            image_generation.callback = self.step_progress_update

            send_node = self.node_graph.get_node_by_name("image_send")
            send_node.image_callback = self.preview_image
        else:
            changed = self.image_generation_data.get_changed_attributes()
            for attr_name, new_value in changed.items():
                node = self.node_graph.get_node_by_name(attr_name)

                if attr_name == "model":
                    node.update_model(
                        path=new_value["path"],
                        model_name=new_value["name"],
                        version=new_value["version"],
                        model_type=new_value["type"],
                    )
                elif attr_name == "vae":
                    node.update_model(
                        path=new_value["path"], vae_name=new_value["name"]
                    )
                else:
                    node.update_value(new_value)

            self.image_generation_data.update_previous_state()

        if self.node_graph.sequential_offload != self.sequential_offload:
            self.check_and_update(
                "sequential_offload", "sequential_offload", self.sequential_offload
            )
        elif self.node_graph.cpu_offload != self.model_offload:
            self.check_and_update("cpu_offload", "model_offload", self.model_offload)

        # process loras
        if len(self.lora_list.loras) > 0:
            sdxl_model = self.node_graph.get_node_by_name("model")
            image_generation = self.node_graph.get_node_by_name("image_generation")
            lora_scale = self.node_graph.get_node_by_name("lora_scale")
            prompts_encoder = self.node_graph.get_node_by_name("prompts_encoder")
            decoder = self.node_graph.get_node_by_name("decoder")
            image_send = self.node_graph.get_node_by_name("image_send")

            # if there's a image dropped to generate, reset all the loras since its impossible
            # to keep track of the ids for the nodes
            if self.lora_list.dropped_image:
                sdxl_model.unload_lora_weights()
                lora_nodes = self.node_graph.get_all_nodes_class(LoraNode)
                for lora_node in lora_nodes:
                    self.node_graph.delete_node_by_id(lora_node.id)

                new_loras = self.lora_list.loras

                for lora in new_loras:
                    lora_node = LoraNode(
                        path=lora.path,
                        adapter_name=lora.filename,
                        scale=lora.weight,
                        lora_name=lora.name,
                        version=lora.version,
                    )
                    lora_node.connect("unet", sdxl_model, "unet")
                    lora_node.connect("text_encoder_1", sdxl_model, "text_encoder_1")
                    lora_node.connect("text_encoder_2", sdxl_model, "text_encoder_2")
                    lora_node.connect("global_lora_scale", lora_scale, "value")
                    self.node_graph.add_node(lora_node, lora.filename)
                    lora.id = lora_node.id
                    image_generation.connect("lora", lora_node, "lora")

                    # this is manually updated since it doesn't have a relation with the node (add a system for this)
                    prompts_encoder.updated = True

                    # ugly patch while I find why they dont get flagged as updated
                    decoder.updated = True
                    image_send.updated = True

            else:
                new_loras = self.lora_list.get_added()

                if len(new_loras) > 0:
                    for lora in new_loras:
                        lora_node = LoraNode(
                            path=lora.path,
                            adapter_name=lora.filename,
                            scale=lora.weight,
                            lora_name=lora.name,
                            version=lora.version,
                        )
                        lora_node.connect("unet", sdxl_model, "unet")
                        lora_node.connect(
                            "text_encoder_1", sdxl_model, "text_encoder_1"
                        )
                        lora_node.connect(
                            "text_encoder_2", sdxl_model, "text_encoder_2"
                        )
                        lora_node.connect("global_lora_scale", lora_scale, "value")
                        self.node_graph.add_node(lora_node, lora.filename)
                        lora.id = lora_node.id
                        image_generation.connect("lora", lora_node, "lora")

                        # this is manually updated since it doesn't have a relation with the node (add a system for this)
                        prompts_encoder.updated = True

                        # ugly patch while I find why they dont get flagged as updated
                        decoder.updated = True
                        image_send.updated = True

                modified_loras = self.lora_list.get_modified()

                if len(modified_loras) > 0:
                    for lora in modified_loras:
                        lora_node = self.node_graph.get_node(lora.id)
                        lora_node.update_scale(lora.weight)

                        # same as before
                        prompts_encoder.updated = True
                        decoder.updated = True
                        image_send.updated = True

                removed_loras = self.lora_list.get_removed()

                if len(removed_loras) > 0:
                    adapter_names = []
                    for lora in removed_loras:
                        self.node_graph.delete_node_by_id(lora.id)
                        adapter_names.append(lora.filename)

                    if len(adapter_names) > 0:
                        sdxl_model.delete_adapters(adapter_names)

                    # same as before
                    prompts_encoder.updated = True
                    decoder.updated = True
                    image_send.updated = True
        else:
            removed_loras = self.lora_list.get_removed()

            if len(removed_loras) > 0:
                for lora in removed_loras:
                    self.node_graph.delete_node_by_id(lora.id)

                sdxl_model.unload_lora_weights()

                # same as before
                prompts_encoder.updated = True
                decoder.updated = True
                image_send.updated = True

        self.lora_list.save_state()
        self.lora_list.dropped_image = False

        try:
            self.node_graph()
        except KeyError:
            self.generation_error.emit("There was an error while generating.", False)

        if not self.node_graph.updated:
            self.generation_error.emit("Nothing was changed", False)

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)

    def preview_image(self, image):
        self.generation_finished.emit(image)

    def reset_model_path(self, model_name):
        model_node = self.node_graph.get_node_by_name(model_name)
        if model_node is not None:
            # model_node.path = ""
            model_node.set_updated()

    def check_and_update(self, attr1, attr2, value):
        if getattr(self.node_graph, attr1) != getattr(self, attr2):
            self.reset_model_path("model")
            self.reset_model_path("vae_model")
            setattr(self.node_graph, attr1, value)
