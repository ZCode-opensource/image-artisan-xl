from typing import List

from PyQt6.QtCore import QObject

from iartisanxl.generation.lora_data_object import LoraDataObject
from iartisanxl.generation.model_data_object import ModelDataObject
from iartisanxl.generation.vae_data_object import VaeDataObject


class ImageGenData(QObject):
    __slots__ = (
        "_module",
        "_seed",
        "_image_width",
        "_image_height",
        "_steps",
        "_guidance",
        "_base_scheduler",
        "_lora_scale",
        "_loras",
        "_model",
        "_vae",
        "_positive_prompt_clipl",
        "_positive_prompt_clipg",
        "_negative_prompt_clipl",
        "_negative_prompt_clipg",
        "_clip_skip",
    )

    def __init__(
        self,
        *,
        module: str,
        seed: int,
        image_width: int,
        image_height: int,
        steps: int,
        guidance: float,
        base_scheduler: int,
        lora_scale: float,
        loras: List[LoraDataObject],
        model: ModelDataObject,
        vae: VaeDataObject,
        positive_prompt_clipl: str,
        positive_prompt_clipg: str,
        negative_prompt_clipl: str,
        negative_prompt_clipg: str,
        clip_skip: int,
    ):
        super().__init__()
        self._module = module
        self._seed = seed
        self._image_width = image_width
        self._image_height = image_height
        self._steps = steps
        self._guidance = guidance
        self._base_scheduler = base_scheduler
        self._lora_scale = lora_scale
        self._loras = loras
        self._model = model
        self._vae = vae
        self._positive_prompt_clipl = positive_prompt_clipl
        self._positive_prompt_clipg = positive_prompt_clipg
        self._negative_prompt_clipl = negative_prompt_clipl
        self._negative_prompt_clipg = negative_prompt_clipg
        self._clip_skip = clip_skip

    @property
    def module(self) -> str:
        return self._module

    @module.setter
    def module(self, value: str) -> None:
        self._module = value

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = value

    @property
    def image_width(self) -> int:
        return self._image_width

    @image_width.setter
    def image_width(self, value: int) -> None:
        self._image_width = value

    @property
    def image_height(self) -> int:
        return self._image_height

    @image_height.setter
    def image_height(self, value: int) -> None:
        self._image_height = value

    @property
    def steps(self) -> int:
        return self._steps

    @steps.setter
    def steps(self, value: int) -> None:
        self._steps = value

    @property
    def guidance(self) -> float:
        return self._guidance

    @guidance.setter
    def guidance(self, value: float) -> None:
        self._guidance = value

    @property
    def base_scheduler(self) -> int:
        return self._base_scheduler

    @base_scheduler.setter
    def base_scheduler(self, value: int) -> None:
        self._base_scheduler = value

    @property
    def model(self) -> ModelDataObject:
        return self._model

    @model.setter
    def model(self, value: ModelDataObject) -> None:
        self._model = value

    @property
    def vae(self) -> VaeDataObject:
        return self._vae

    @vae.setter
    def vae(self, value: VaeDataObject) -> None:
        self._vae = value

    @property
    def positive_prompt_clipl(self) -> str:
        return self._positive_prompt_clipl

    @positive_prompt_clipl.setter
    def positive_prompt_clipl(self, value: str) -> None:
        self._positive_prompt_clipl = value

    @property
    def positive_prompt_clipg(self) -> str:
        return self._positive_prompt_clipg

    @positive_prompt_clipg.setter
    def positive_prompt_clipg(self, value: str) -> None:
        self._positive_prompt_clipg = value

    @property
    def negative_prompt_clipl(self) -> str:
        return self._negative_prompt_clipl

    @negative_prompt_clipl.setter
    def negative_prompt_clipl(self, value: str) -> None:
        self._negative_prompt_clipl = value

    @property
    def negative_prompt_clipg(self) -> str:
        return self._negative_prompt_clipg

    @negative_prompt_clipg.setter
    def negative_prompt_clipg(self, value: str) -> None:
        self._negative_prompt_clipg = value

    @property
    def clip_skip(self) -> int:
        return self._clip_skip

    @clip_skip.setter
    def clip_skip(self, value: int) -> None:
        self._clip_skip = value

    def update_attributes(self, data: dict):
        error = None
        for key, value in data.items():
            if key == "loras":
                for lora_data in value:
                    try:
                        lora = LoraDataObject(**lora_data)
                        self.add_lora(lora)
                    except TypeError:
                        error = "LoRA information is not compatible."
            elif key == "model":
                try:
                    model = ModelDataObject(**value)
                    self._model = model
                except TypeError:
                    self._model = None
                    error = "Model information is not compatible."
            elif key == "vae":
                try:
                    vae = VaeDataObject(**value)
                except TypeError:
                    vae = VaeDataObject(name="Model default", path="")
                    error = "Vae information is not compatible, using model default."
                self._vae = vae
            else:
                attr = f"_{key}"
                if attr in self.__slots__:
                    setattr(self, attr, value)

        return error

    @property
    def lora_scale(self) -> float:
        return self._lora_scale

    @lora_scale.setter
    def lora_scale(self, value: float) -> None:
        self._lora_scale = value

    @property
    def loras(self) -> List[LoraDataObject]:
        return self._loras

    @loras.setter
    def loras(self, value: List[LoraDataObject]):
        self._loras = value

    def add_lora(self, lora: LoraDataObject):
        self._loras.append(lora)

    def remove_lora(self, lora_to_remove: LoraDataObject):
        for index, lora in enumerate(self._loras):
            if lora == lora_to_remove:
                del self._loras[index]
                break

    def copy(self):
        loras_copy = [lora.copy() for lora in self.loras]

        new_obj = ImageGenData(
            module=self.module,
            seed=self.seed,
            image_width=self.image_width,
            image_height=self.image_height,
            steps=self.steps,
            guidance=self.guidance,
            base_scheduler=self.base_scheduler,
            lora_scale=self.lora_scale,
            loras=loras_copy,
            model=self.model.copy(),
            vae=self.vae,
            positive_prompt_clipl=self.positive_prompt_clipl,
            positive_prompt_clipg=self.positive_prompt_clipg,
            negative_prompt_clipl=self.negative_prompt_clipl,
            negative_prompt_clipg=self.negative_prompt_clipg,
            clip_skip=self._clip_skip,
        )
        return new_obj
