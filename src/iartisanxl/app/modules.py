from importlib.resources import files

from iartisanxl.modules.text.text_to_image_module import TextToImageModule
from iartisanxl.modules.dataset.dataset_module import DatasetModule
from iartisanxl.modules.image.image_to_image_module import ImageToImageModule
from iartisanxl.modules.train.lora.train_lora_module import TrainLoraModule

TXT2IMG_ICON = files("iartisanxl.theme.icons").joinpath("txtimg.png")
IMG2IMG_ICON = files("iartisanxl.theme.icons").joinpath("imgtoimg.png")
INPAINT_ICON = files("iartisanxl.theme.icons").joinpath("inpainting.png")
GALLERY_ICON = files("iartisanxl.theme.icons").joinpath("gallery.png")
CANVAS_ICON = files("iartisanxl.theme.icons").joinpath("canvas.png")
NODE_ICON = files("iartisanxl.theme.icons").joinpath("nodes.png")
FINETUNE_ICON = files("iartisanxl.theme.icons").joinpath("finetune.png")
LORA_ICON = files("iartisanxl.theme.icons").joinpath("train_lora.png")
DATASET_ICON = files("iartisanxl.theme.icons").joinpath("dataset.png")

MODULES = {
    "Text to image": (TXT2IMG_ICON, TextToImageModule),
    # "Image to image": (IMG2IMG_ICON, ImageToImageModule),
    # "Inpainting": (INPAINT_ICON, TextToImageModule),
    # "Gallery": (GALLERY_ICON, TextToImageModule),
    # "Canvas": (CANVAS_ICON, TextToImageModule),
    # "Nodes": (NODE_ICON, TextToImageModule),
    # "Finetune model": (FINETUNE_ICON, TextToImageModule),
    # "Train LoRA": (LORA_ICON, TrainLoraModule),
    # "Dataset": (DATASET_ICON, DatasetModule),
}
