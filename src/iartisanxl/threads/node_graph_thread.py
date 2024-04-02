import logging
import os

import torch
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanxl.app.directories import DirectoriesObject
from iartisanxl.generation.adapter_list import AdapterList
from iartisanxl.generation.image_generation_data import ImageGenerationData
from iartisanxl.graph.iartisan_node_error import IArtisanNodeError
from iartisanxl.graph.iartisanxl_node_graph import ImageArtisanNodeGraph
from iartisanxl.graph.nodes.controlnet_model_node import ControlnetModelNode
from iartisanxl.graph.nodes.controlnet_node import ControlnetNode
from iartisanxl.graph.nodes.image_encoder_model_node import ImageEncoderModelNode
from iartisanxl.graph.nodes.image_load_node import ImageLoadNode
from iartisanxl.graph.nodes.ip_adapter_merge_node import IPAdapterMergeNode
from iartisanxl.graph.nodes.ip_adapter_model_node import IPAdapterModelNode
from iartisanxl.graph.nodes.ip_adapter_node import IPAdapterNode
from iartisanxl.graph.nodes.lora_node import LoraNode
from iartisanxl.graph.nodes.t2i_adapter_model_node import T2IAdapterModelNode
from iartisanxl.graph.nodes.t2i_adapter_node import T2IAdapterNode
from iartisanxl.modules.common.lora.lora_list import LoraList


controlnet_dict = {
    "controlnet_canny_model": "controlnet-canny-sdxl-1.0-small",
    "controlnet_depth_model": "controlnet-depth-sdxl-1.0-small",
    "controlnet_inpaint_model": "controlnet-inpaint-dreamer-sdxl",
}

t2i_adapter_dict = {
    "t2i_adapter_canny_model": "t2i-adapter-canny-sdxl-1.0",
    "t2i_adapter_depth_model": "t2i-adapter-depth-midas-sdxl-1.0",
    "t2i_adapter_lineart_model": "t2i-adapter-lineart-sdxl-1.0",
    "t2i_adapter_sketch_model": "t2i-adapter-sketch-sdxl-1.0",
    "t2i-recolor": "t2i-recolor",
}


ip_adapter_dict = {
    "ip_adapter_vit_h": "ip-adapter_sdxl_vit-h.safetensors",
    "ip_adapter_plus": "ip-adapter-plus_sdxl_vit-h.safetensors",
    "ip_adapter_plus_face": "ip-adapter-plus-face_sdxl_vit-h.safetensors",
    "ip_plus_composition_sdxl": "ip_plus_composition_sdxl.safetensors",
}


class NodeGraphThread(QThread):
    status_changed = pyqtSignal(str)
    progress_update = pyqtSignal(int, torch.Tensor)
    generation_finished = pyqtSignal(Image.Image)
    generation_error = pyqtSignal(str, bool)
    generation_aborted = pyqtSignal()

    def __init__(
        self,
        directories: DirectoriesObject = None,
        node_graph: ImageArtisanNodeGraph = None,
        image_generation_data: ImageGenerationData = None,
        lora_list: LoraList = None,
        controlnet_list: AdapterList = None,
        t2i_adapter_list: AdapterList = None,
        ip_adapter_list: AdapterList = None,
        model_offload: bool = False,
        sequential_offload: bool = False,
        torch_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.logger = logging.getLogger()
        self.directories = directories
        self.node_graph = node_graph
        self.image_generation_data = image_generation_data
        self.lora_list = lora_list
        self.controlnet_list = controlnet_list
        self.t2i_adapter_list = t2i_adapter_list
        self.ip_adapter_list = ip_adapter_list
        self.model_offload = model_offload
        self.sequential_offload = sequential_offload
        self.torch_dtype = torch_dtype

    def run(self):  # noqa: C901
        self.node_graph.torch_dtype = self.torch_dtype

        self.status_changed.emit("Generating image...")
        if self.node_graph.node_counter == 0:
            self.node_graph = self.image_generation_data.create_text_to_image_graph(self.node_graph)
        else:
            changed = self.image_generation_data.get_changed_attributes()

            for attr_name, new_value in changed.items():
                node = self.node_graph.get_node_by_name(attr_name)

                if node is not None:
                    if attr_name == "model":
                        node.update_model(
                            path=new_value["path"],
                            model_name=new_value["name"],
                            version=new_value["version"],
                            model_type=new_value["type"],
                        )
                    elif attr_name == "vae":
                        node.update_model(path=new_value["path"], vae_name=new_value["name"])
                    else:
                        node.update_value(new_value)

        # connect the essential callbacks
        self.node_graph.set_abort_function(self.on_aborted)
        image_generation = self.node_graph.get_node_by_name("image_generation")
        image_generation.callback = self.step_progress_update
        send_node = self.node_graph.get_node_by_name("image_send")
        send_node.image_callback = self.preview_image

        self.image_generation_data.update_previous_state()

        if self.node_graph.sequential_offload != self.sequential_offload:
            self.check_and_update("sequential_offload", "sequential_offload", self.sequential_offload)
        elif self.node_graph.cpu_offload != self.model_offload:
            self.check_and_update("cpu_offload", "model_offload", self.model_offload)

        sdxl_model = self.node_graph.get_node_by_name("model")
        prompts_encoder = self.node_graph.get_node_by_name("prompts_encoder")
        image_generation = self.node_graph.get_node_by_name("image_generation")

        # process loras
        lora_scale = self.node_graph.get_node_by_name("lora_scale")

        # if there's a image dropped to generate, reset all the loras since I'm too lazy to
        # keep track of all the ids for the nodes
        if self.lora_list.dropped_image:
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
                lora.node_id = lora_node.id
                image_generation.connect("lora", lora_node, "lora")
                prompts_encoder.connect("lora", lora_node, "lora")
        else:
            removed_loras = self.lora_list.get_removed()

            if len(removed_loras) > 0:
                for lora in removed_loras:
                    self.node_graph.delete_node_by_id(lora.node_id)

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
                    lora_node.connect("text_encoder_1", sdxl_model, "text_encoder_1")
                    lora_node.connect("text_encoder_2", sdxl_model, "text_encoder_2")
                    lora_node.connect("global_lora_scale", lora_scale, "value")
                    self.node_graph.add_node(lora_node, lora.filename)
                    lora.node_id = lora_node.id
                    image_generation.connect("lora", lora_node, "lora")
                    prompts_encoder.connect("lora", lora_node, "lora")

            modified_loras = self.lora_list.get_modified()

            if len(modified_loras) > 0:
                for lora in modified_loras:
                    lora_node = self.node_graph.get_node(lora.node_id)

                    if lora_node is not None:
                        lora_node.update_lora(lora.weight, lora.enabled)

        self.lora_list.save_state()
        self.lora_list.dropped_image = False

        # process controlnets
        controlnet_types = self.controlnet_list.get_used_types()

        for controlnet_type in controlnet_types:
            self.get_controlnet_model(controlnet_type)

        if len(self.controlnet_list.adapters) > 0:
            added_controlnets = self.controlnet_list.get_added()

            if len(added_controlnets) > 0:
                for controlnet in added_controlnets:
                    controlnet_image_node = ImageLoadNode(path=controlnet.preprocessor_image)
                    controlnet_node = ControlnetNode(
                        controlnet.type_index,
                        controlnet.adapter_type,
                        controlnet.conditioning_scale,
                        controlnet.guidance_start,
                        controlnet.guidance_end,
                    )

                    controlnet_model_node = self.get_controlnet_model(controlnet.adapter_type)
                    controlnet_node.connect("controlnet_model", controlnet_model_node, "controlnet_model")
                    controlnet_node.connect("image", controlnet_image_node, "image")
                    image_generation.connect("controlnet", controlnet_node, "controlnet")
                    self.node_graph.add_node(controlnet_node)
                    controlnet.node_id = controlnet_node.id
                    controlnet_node.name = f"controlnet_{controlnet.adapter_type}_{controlnet_node.id}"
                    self.node_graph.add_node(controlnet_image_node, f"control_image_{controlnet_node.id}")

            modified_controlnets = self.controlnet_list.get_modified()

            if len(modified_controlnets) > 0:
                for controlnet in modified_controlnets:
                    controlnet_node = self.node_graph.get_node(controlnet.node_id)

                    if controlnet.type_index != controlnet_node.type_index:
                        # disconnect old model
                        controlnet_model_node = self.get_controlnet_model(controlnet_node.adapter_type)
                        controlnet_node.disconnect("controlnet_model", controlnet_model_node, "controlnet_model")

                        # connect new changed model
                        controlnet_model_node = self.get_controlnet_model(controlnet.adapter_type)
                        controlnet_node.connect("controlnet_model", controlnet_model_node, "controlnet_model")

                    # update rest of params
                    controlnet_node.update_controlnet(
                        controlnet.type_index,
                        controlnet.adapter_type,
                        controlnet.enabled,
                        controlnet.conditioning_scale,
                        controlnet.guidance_start,
                        controlnet.guidance_end,
                    )

                    # update image
                    control_image_node = self.node_graph.get_node_by_name(f"control_image_{controlnet.node_id}")
                    control_image_node.update_path(controlnet.preprocessor_image)

        removed_controlnets = self.controlnet_list.get_removed()
        if len(removed_controlnets) > 0:
            for controlnet in removed_controlnets:
                control_image_node = self.node_graph.get_node_by_name(f"control_image_{controlnet.node_id}")
                self.node_graph.delete_node_by_id(control_image_node.id)
                self.node_graph.delete_node_by_id(controlnet.node_id)

        self.controlnet_list.save_state()
        self.controlnet_list.dropped_image = False

        # process t2i_adapters
        t2i_adapter_types = self.t2i_adapter_list.get_used_types()

        for t2i_adapter_type in t2i_adapter_types:
            self.get_t2i_adapter_model(t2i_adapter_type)

        if len(self.t2i_adapter_list.adapters) > 0:
            added_t2i_adapters = self.t2i_adapter_list.get_added()

            if len(added_t2i_adapters) > 0:
                for t2i_adapter in added_t2i_adapters:
                    t2i_adapter_image_node = ImageLoadNode(path=t2i_adapter.preprocessor_image)
                    t2i_adapter_node = T2IAdapterNode(
                        t2i_adapter.type_index,
                        t2i_adapter.adapter_type,
                        t2i_adapter.conditioning_scale,
                        t2i_adapter.conditioning_factor,
                    )

                    t2i_adapter_model_node = self.get_t2i_adapter_model(t2i_adapter.adapter_type)
                    t2i_adapter_node.connect("t2i_adapter_model", t2i_adapter_model_node, "t2i_adapter_model")
                    t2i_adapter_node.connect("image", t2i_adapter_image_node, "image")
                    image_generation.connect("t2i_adapter", t2i_adapter_node, "t2i_adapter")
                    self.node_graph.add_node(t2i_adapter_node)
                    t2i_adapter.node_id = t2i_adapter_node.id
                    t2i_adapter_node.name = f"t2i_adapter{t2i_adapter.adapter_type}_{t2i_adapter_node.id}"
                    self.node_graph.add_node(t2i_adapter_image_node, f"adapter_image_{t2i_adapter_node.id}")

            modified_t2i_adapters = self.t2i_adapter_list.get_modified()

            if len(modified_t2i_adapters) > 0:
                for t2i_adapter in modified_t2i_adapters:
                    t2i_adapter_node = self.node_graph.get_node(t2i_adapter.node_id)

                    if t2i_adapter.type_index != t2i_adapter_node.type_index:
                        # disconnect old model
                        t2i_adapter_model_node = self.get_t2i_adapter_model(t2i_adapter_node.adapter_type)
                        t2i_adapter_node.disconnect("t2i_adapter_model", t2i_adapter_model_node, "t2i_adapter_model")

                        # connect new changed model
                        t2i_adapter_model_node = self.get_t2i_adapter_model(t2i_adapter.adapter_type)
                        t2i_adapter_node.connect("t2i_adapter_model", t2i_adapter_model_node, "t2i_adapter_model")

                    # update rest of params
                    t2i_adapter_node.update_adapter(
                        t2i_adapter.type_index,
                        t2i_adapter.adapter_type,
                        t2i_adapter.enabled,
                        t2i_adapter.conditioning_scale,
                        t2i_adapter.conditioning_factor,
                    )

                    # update image
                    t2i_adapter_image_node = self.node_graph.get_node_by_name(f"adapter_image_{t2i_adapter.node_id}")
                    t2i_adapter_image_node.update_path(t2i_adapter.preprocessor_image)

        removed_t2i_adapters = self.t2i_adapter_list.get_removed()
        if len(removed_t2i_adapters) > 0:
            for t2i_adapter in removed_t2i_adapters:
                adapter_image_node = self.node_graph.get_node_by_name(f"adapter_image_{t2i_adapter.node_id}")
                self.node_graph.delete_node_by_id(adapter_image_node.id)
                self.node_graph.delete_node_by_id(t2i_adapter.node_id)

        self.t2i_adapter_list.save_state()
        self.t2i_adapter_list.dropped_image = False

        # Process IP adapters
        ip_adapter_types = self.ip_adapter_list.get_used_types()

        for _ in ip_adapter_types:
            image_encoder_h_model_node = self.node_graph.get_node_by_name("ip_image_encoder_h")

            if image_encoder_h_model_node is None:
                image_encoder_h_model_node = ImageEncoderModelNode(
                    path=os.path.join(self.directories.models_ip_adapters, "image_encoder")
                )
                self.node_graph.add_node(image_encoder_h_model_node, "ip_image_encoder_h")

        if len(self.ip_adapter_list.adapters) > 0:
            added_ip_adapters = self.ip_adapter_list.get_added()

            ip_adapter_merge_node = self.node_graph.get_node_by_name("ip_adapter_merge_node")

            if ip_adapter_merge_node is None:
                ip_adapter_merge_node = IPAdapterMergeNode()
                ip_adapter_merge_node.connect("unet", sdxl_model, "unet")
                image_generation.connect("ip_adapter", ip_adapter_merge_node, "ip_adapter")
                self.node_graph.add_node(ip_adapter_merge_node, "ip_adapter_merge_node")

            if len(added_ip_adapters) > 0:
                for ip_adapter in added_ip_adapters:
                    ip_adapter_node = IPAdapterNode(
                        ip_adapter.type_index,
                        ip_adapter.adapter_type,
                        ip_adapter.ip_adapter_scale,
                        ip_adapter.granular_scale_enabled,
                        ip_adapter.granular_scale,
                    )
                    self.node_graph.add_node(ip_adapter_node)
                    ip_adapter.node_id = ip_adapter_node.id

                    ip_adapter_model_node = self.get_ip_adapter_model(ip_adapter.adapter_type)
                    ip_adapter_node.connect("image_encoder", image_encoder_h_model_node, "image_encoder")
                    ip_adapter_node.connect("ip_adapter_model", ip_adapter_model_node, "ip_adapter_model")
                    ip_adapter_merge_node.connect("ip_adapter", ip_adapter_node, "ip_adapter")

                    if ip_adapter.mask_image is not None:
                        ip_adapter_mask_image_node = ImageLoadNode(
                            path=ip_adapter.mask_image.mask_image.image_filename
                        )
                        self.node_graph.add_node(
                            ip_adapter_mask_image_node, f"adapter_mask_image_{ip_adapter_node.id}"
                        )
                        ip_adapter_node.connect("mask_alpha_image", ip_adapter_mask_image_node, "image")

                    for image in ip_adapter.images:
                        ip_adapter_image_node = ImageLoadNode(
                            path=image.image,
                            weight=image.weight,
                            noise=image.noise,
                            noise_index=image.noise_type_index,
                        )
                        self.node_graph.add_node(
                            ip_adapter_image_node, f"adapter_image_{ip_adapter_node.id}_{image.ip_adapter_id}"
                        )
                        image.node_id = ip_adapter_image_node.id
                        ip_adapter_node.connect("image", ip_adapter_image_node, "image")

                    ip_adapter.save_image_state()

            modified_ip_adapters = self.ip_adapter_list.get_modified()

            if len(modified_ip_adapters) > 0:
                for ip_adapter in modified_ip_adapters:
                    ip_adapter_node = self.node_graph.get_node(ip_adapter.node_id)
                    type_changed = False

                    if ip_adapter.type_index != ip_adapter_node.type_index:
                        type_changed = True
                        # disconnect old model
                        ip_adapter_model_node = self.get_ip_adapter_model(ip_adapter_node.adapter_type)
                        ip_adapter_node.disconnect("ip_adapter_model", ip_adapter_model_node, "ip_adapter_model")

                        # connect new changed model
                        ip_adapter_model_node = self.get_ip_adapter_model(ip_adapter.adapter_type)
                        ip_adapter_node.connect("ip_adapter_model", ip_adapter_model_node, "ip_adapter_model")

                    # update rest of params
                    ip_adapter_node.update_adapter(
                        ip_adapter.type_index,
                        ip_adapter.adapter_type,
                        ip_adapter.enabled,
                        ip_adapter.ip_adapter_scale,
                        ip_adapter.granular_scale_enabled,
                        ip_adapter.granular_scale,
                        type_changed,
                    )

                    if ip_adapter.mask_image is not None:
                        ip_adapter_mask_image_node = self.node_graph.get_node_by_name(
                            f"adapter_mask_image_{ip_adapter_node.id}"
                        )

                        if ip_adapter_mask_image_node is None:
                            ip_adapter_mask_image_node = ImageLoadNode(
                                path=ip_adapter.mask_image.mask_image.image_filename
                            )
                            self.node_graph.add_node(
                                ip_adapter_mask_image_node, f"adapter_mask_image_{ip_adapter_node.id}"
                            )
                            ip_adapter_node.connect("mask_alpha_image", ip_adapter_mask_image_node, "image")
                        else:
                            ip_adapter_mask_image_node.update_path(ip_adapter.mask_image.mask_image.image_filename)
                    else:
                        self.node_graph.delete_node_by_name(f"adapter_mask_image_{ip_adapter_node.id}")

                    added_images = ip_adapter.get_added_images()
                    modified_images = ip_adapter.get_modified_images()
                    deleted_images = ip_adapter.get_removed_images()

                    if len(added_images) > 0:
                        for image in added_images:
                            ip_adapter_image_node = ImageLoadNode(
                                path=image.image,
                                weight=image.weight,
                                noise=image.noise,
                                noise_index=image.noise_type_index,
                            )
                            self.node_graph.add_node(
                                ip_adapter_image_node, f"adapter_image_{ip_adapter_node.id}_{image.ip_adapter_id}"
                            )
                            image.node_id = ip_adapter_image_node.id
                            ip_adapter_node.connect("image", ip_adapter_image_node, "image")

                    if len(modified_images) > 0:
                        for image in modified_images:
                            ip_adapter_image_node = self.node_graph.get_node(image.node_id)
                            ip_adapter_image_node.update_path_weight_noise(
                                image.image,
                                weight=image.weight,
                                noise=image.noise,
                                noise_index=image.noise_type_index,
                            )

                    if len(deleted_images) > 0:
                        for image in deleted_images:
                            self.node_graph.delete_node_by_id(image.node_id)

                    ip_adapter.save_image_state()
        else:
            ip_adapter_merge_node = self.node_graph.get_node_by_name("ip_adapter_merge_node")
            if ip_adapter_merge_node:
                self.node_graph.delete_node(ip_adapter_merge_node)

        removed_ip_adapters = self.ip_adapter_list.get_removed()
        if len(removed_ip_adapters) > 0:
            for ip_adapter in removed_ip_adapters:
                for image in ip_adapter.images:
                    self.node_graph.delete_node_by_id(image.node_id)

                ip_adapter_node = self.node_graph.get_node(ip_adapter.node_id)
                self.node_graph.delete_node_by_id(ip_adapter.node_id)

        self.ip_adapter_list.save_state()
        self.ip_adapter_list.dropped_image = False

        try:
            self.node_graph()
        except IArtisanNodeError as e:
            self.generation_error.emit(f"Error in node '{e.node_name}': {e}", False)

        if not self.node_graph.updated:
            self.generation_error.emit("Nothing was changed", False)

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)

    def preview_image(self, image):
        self.generation_finished.emit(image)

    def reset_model_path(self, model_name):
        model_node = self.node_graph.get_node_by_name(model_name)
        if model_node is not None:
            model_node.set_updated()

    def check_and_update(self, attr1, attr2, value):
        if getattr(self.node_graph, attr1) != getattr(self, attr2):
            self.reset_model_path("model")
            self.reset_model_path("vae_model")
            setattr(self.node_graph, attr1, value)

    def abort_graph(self):
        self.node_graph.abort_graph()

    def on_aborted(self):
        self.generation_aborted.emit()

    def clean_up(self):
        self.node_graph = None
        self.logger = None
        self.directories = None
        self.image_generation_data = None
        self.lora_list = None
        self.controlnet_list = None
        self.t2i_adapter_list = None

    def get_controlnet_model(self, controlnet_type):
        controlnet_model_node = self.node_graph.get_node_by_name(controlnet_type)

        if controlnet_model_node is None:
            controlnet_model_path = controlnet_dict.get(controlnet_type, "")

            controlnet_model_node = ControlnetModelNode(
                path=os.path.join(self.directories.models_controlnets, controlnet_model_path)
            )
            self.node_graph.add_node(controlnet_model_node, controlnet_type)

        return controlnet_model_node

    def get_t2i_adapter_model(self, adapter_type):
        t2i_adapter_model_node = self.node_graph.get_node_by_name(adapter_type)

        if t2i_adapter_model_node is None:
            t2i_adapter_model_path = t2i_adapter_dict.get(adapter_type, "")

            t2i_adapter_model_node = T2IAdapterModelNode(
                path=os.path.join(self.directories.models_t2i_adapters, t2i_adapter_model_path)
            )
            self.node_graph.add_node(t2i_adapter_model_node, adapter_type)

        return t2i_adapter_model_node

    def get_ip_adapter_model(self, ip_adapter_type):
        ip_adapter_model_node = self.node_graph.get_node_by_name(ip_adapter_type)

        if ip_adapter_model_node is None:
            ip_adapter_model_file = ip_adapter_dict.get(ip_adapter_type, "")

            ip_adapter_model_node = IPAdapterModelNode(
                path=os.path.join(self.directories.models_ip_adapters, ip_adapter_type, ip_adapter_model_file)
            )
            self.node_graph.add_node(ip_adapter_model_node, ip_adapter_type)

        return ip_adapter_model_node
