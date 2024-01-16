import attr
import unittest


from iartisanxl.generation.image_generation_data import ImageGenerationData
from iartisanxl.generation.model_data_object import ModelDataObject
from iartisanxl.generation.vae_data_object import VaeDataObject
from iartisanxl.graph.iartisanxl_node_graph import ImageArtisanNodeGraph


class TestImageGenerationData(unittest.TestCase):
    def test_init(self):
        obj = ImageGenerationData(
            module="test_module",
            seed=123,
            image_width=800,
            image_height=600,
            steps=10,
            guidance=5.0,
            base_scheduler=1,
            lora_scale=0.5,
            model=ModelDataObject(name="test_model", path="/path/to/model"),
            vae=VaeDataObject(name="test_vae", path="/path/to/vae"),
            positive_prompt_clipl="test_positive_clipl",
            positive_prompt_clipg="test_positive_clipg",
            negative_prompt_clipl="test_negative_clipl",
            negative_prompt_clipg="test_negative_clipg",
            clip_skip=2,
        )

        self.assertEqual(obj.module, "test_module")
        self.assertEqual(obj.seed, 123)
        self.assertEqual(obj.image_width, 800)
        self.assertEqual(obj.image_height, 600)
        self.assertEqual(obj.steps, 10)
        self.assertEqual(obj.guidance, 5.0)
        self.assertEqual(obj.base_scheduler, 1)
        self.assertEqual(obj.lora_scale, 0.5)
        self.assertEqual(obj.model.name, "test_model")
        self.assertEqual(obj.model.path, "/path/to/model")
        self.assertEqual(obj.vae.name, "test_vae")
        self.assertEqual(obj.vae.path, "/path/to/vae")
        self.assertEqual(obj.positive_prompt_clipl, "test_positive_clipl")
        self.assertEqual(obj.positive_prompt_clipg, "test_positive_clipg")
        self.assertEqual(obj.negative_prompt_clipl, "test_negative_clipl")
        self.assertEqual(obj.negative_prompt_clipg, "test_negative_clipg")
        self.assertEqual(obj.clip_skip, 2)

    def test_update_previous_state(self):
        obj = ImageGenerationData()
        original_state = attr.asdict(obj)
        original_state.pop("previous_state", None)
        obj.update_previous_state()
        self.assertEqual(obj.previous_state, original_state)

    def test_get_changed_attributes(self):
        obj = ImageGenerationData(module="test_module")
        obj.update_previous_state()
        obj.module = "new_test_module"
        obj.image_width = 1152
        changed_attributes = obj.get_changed_attributes()
        self.assertEqual(changed_attributes, {"module": "new_test_module", "image_width": 1152})
        obj.update_previous_state()
        obj.image_width = 1024
        obj.image_height = 1152
        changed_attributes = obj.get_changed_attributes()
        self.assertEqual(changed_attributes, {"image_height": 1152, "image_width": 1024})

    def test_create_text_to_image_graph(self):
        obj = ImageGenerationData(
            module="test_module",
            seed=123,
            image_width=800,
            image_height=600,
            steps=10,
            guidance=5.0,
            base_scheduler=1,
            lora_scale=0.5,
            model=ModelDataObject(
                name="test_model",
                path="/path/to/model",
                version="1.0",
                type="diffusers",
            ),
            vae=VaeDataObject(name="test_vae", path="/path/to/vae"),
            positive_prompt_clipl="test_positive_clipl",
            positive_prompt_clipg="test_positive_clipg",
            negative_prompt_clipl="test_negative_clipl",
            negative_prompt_clipg="test_negative_clipg",
            clip_skip=2,
        )

        graph = ImageArtisanNodeGraph()
        obj.create_text_to_image_graph(graph)
        json_graph = graph.to_json()
        valid_json = '{"nodes": [{"class": "NumberNode", "id": 0, "name": "lora_scale", "number": 0.5}, {"class": "NumberNode", "id": 1, "name": "clip_skip", "number": 2}, {"class": "StableDiffusionXLModelNode", "id": 2, "name": "model", "path": "/path/to/model", "model_name": "test_model", "version": "1.0", "model_type": "diffusers"}, {"class": "TextNode", "id": 3, "name": "positive_prompt_clipg", "text": "test_positive_clipg"}, {"class": "TextNode", "id": 4, "name": "positive_prompt_clipl", "text": "test_positive_clipl"}, {"class": "TextNode", "id": 5, "name": "negative_prompt_clipg", "text": "test_negative_clipg"}, {"class": "TextNode", "id": 6, "name": "negative_prompt_clipl", "text": "test_negative_clipl"}, {"class": "PromptsEncoderNode", "id": 7, "name": "prompts_encoder"}, {"class": "VaeModelNode", "id": 8, "name": "vae", "path": "/path/to/vae", "vae_name": "test_vae"}, {"class": "NumberNode", "id": 9, "name": "seed", "number": 123}, {"class": "NumberNode", "id": 10, "name": "image_width", "number": 800}, {"class": "NumberNode", "id": 11, "name": "image_height", "number": 600}, {"class": "LatentsNode", "id": 12, "name": "latents"}, {"class": "SchedulerNode", "id": 13, "name": "base_scheduler", "scheduler_index": 1}, {"class": "NumberNode", "id": 14, "name": "steps", "number": 10}, {"class": "NumberNode", "id": 15, "name": "guidance", "number": 5.0}, {"class": "ImageGenerationNode", "id": 16, "name": "image_generation", "callback": null}, {"class": "LatentsDecoderNode", "id": 17, "name": "decoder"}, {"class": "ImageSendNode", "id": 18, "name": "image_send", "image_callback": null}], "connections": [{"from_node_id": 2, "from_output_name": "tokenizer_1", "to_node_id": 7, "to_input_name": "tokenizer_1"}, {"from_node_id": 2, "from_output_name": "tokenizer_2", "to_node_id": 7, "to_input_name": "tokenizer_2"}, {"from_node_id": 2, "from_output_name": "text_encoder_1", "to_node_id": 7, "to_input_name": "text_encoder_1"}, {"from_node_id": 2, "from_output_name": "text_encoder_2", "to_node_id": 7, "to_input_name": "text_encoder_2"}, {"from_node_id": 3, "from_output_name": "value", "to_node_id": 7, "to_input_name": "positive_prompt_1"}, {"from_node_id": 4, "from_output_name": "value", "to_node_id": 7, "to_input_name": "positive_prompt_2"}, {"from_node_id": 5, "from_output_name": "value", "to_node_id": 7, "to_input_name": "negative_prompt_1"}, {"from_node_id": 6, "from_output_name": "value", "to_node_id": 7, "to_input_name": "negative_prompt_2"}, {"from_node_id": 0, "from_output_name": "value", "to_node_id": 7, "to_input_name": "global_lora_scale"}, {"from_node_id": 9, "from_output_name": "value", "to_node_id": 12, "to_input_name": "seed"}, {"from_node_id": 2, "from_output_name": "num_channels_latents", "to_node_id": 12, "to_input_name": "num_channels_latents"}, {"from_node_id": 10, "from_output_name": "value", "to_node_id": 12, "to_input_name": "width"}, {"from_node_id": 11, "from_output_name": "value", "to_node_id": 12, "to_input_name": "height"}, {"from_node_id": 8, "from_output_name": "vae_scale_factor", "to_node_id": 12, "to_input_name": "vae_scale_factor"}, {"from_node_id": 13, "from_output_name": "scheduler", "to_node_id": 16, "to_input_name": "scheduler"}, {"from_node_id": 14, "from_output_name": "value", "to_node_id": 16, "to_input_name": "num_inference_steps"}, {"from_node_id": 12, "from_output_name": "latents", "to_node_id": 16, "to_input_name": "latents"}, {"from_node_id": 7, "from_output_name": "pooled_prompt_embeds", "to_node_id": 16, "to_input_name": "pooled_prompt_embeds"}, {"from_node_id": 10, "from_output_name": "value", "to_node_id": 16, "to_input_name": "width"}, {"from_node_id": 11, "from_output_name": "value", "to_node_id": 16, "to_input_name": "height"}, {"from_node_id": 7, "from_output_name": "prompt_embeds", "to_node_id": 16, "to_input_name": "prompt_embeds"}, {"from_node_id": 15, "from_output_name": "value", "to_node_id": 16, "to_input_name": "guidance_scale"}, {"from_node_id": 7, "from_output_name": "negative_prompt_embeds", "to_node_id": 16, "to_input_name": "negative_prompt_embeds"}, {"from_node_id": 7, "from_output_name": "negative_pooled_prompt_embeds", "to_node_id": 16, "to_input_name": "negative_pooled_prompt_embeds"}, {"from_node_id": 2, "from_output_name": "unet", "to_node_id": 16, "to_input_name": "unet"}, {"from_node_id": 12, "from_output_name": "generator", "to_node_id": 16, "to_input_name": "generator"}, {"from_node_id": 8, "from_output_name": "vae_scale_factor", "to_node_id": 16, "to_input_name": "vae_scale_factor"}, {"from_node_id": 8, "from_output_name": "vae", "to_node_id": 17, "to_input_name": "vae"}, {"from_node_id": 16, "from_output_name": "latents", "to_node_id": 17, "to_input_name": "latents"}, {"from_node_id": 17, "from_output_name": "image", "to_node_id": 18, "to_input_name": "image"}]}'
        self.assertEqual(json_graph, valid_json)
