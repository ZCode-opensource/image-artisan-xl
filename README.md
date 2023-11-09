# Image Artisan XL

Image Artisan XL is the ultimate desktop application for creating amazing images with the power of artificial intelligence.

<p align="center">
<img src="https://raw.githubusercontent.com/ZCode-opensource/image-artisan-xl/main/src/iartisanxl/theme/images/iartisan_splash.webp" width="350" alt=""/>
<p>

With Image **Artisan XL**, you can unleash your creativity and generate new images from scratch using simple text prompts. Whether you want to create realistic landscapes, fantasy creatures, abstract art, or anything else you can imagine, **Image Artisan XL** has it all. **Image Artisan XL** is powered by Stable Diffusion XL, the best open source image model developed by Stability AI.

No matter what you want to create, **Image Artisan XL** and Stable Diffusion XL will make it happen. Don’t miss this opportunity to get **Image Artisan XL** today and discover the endless possibilities of AI image creation.

## Motivation

I wanted to make a software for Stable Diffusion that doesn’t need web APIs that make you run a separate service and open a web browser, which can be a hassle sometimes.

This software is for the people who like to use desktop apps for editing images and videos, not for the ones who want to automate everything or follow the latest fads in Stable Diffusion. If that’s what you’re looking for, there are other projects that can do that better for you.

Another cool thing about this software is that it has a simple installation process, unlike most other solutions. You just need to download the installer and follow the instructions, and you’re good to go. No need to worry about dependencies, compatibility issues, or complex configurations (only Windows for now).

## Architecture

**Image Artisan XL** is a desktop application that uses PyQt6 as its graphical user interface. It mostly relies on the Diffusers library for generating images, using custom pipelines that are designed to meet the requirements of multithreading and real-time user interface updates.

All the models used in **Image Artisan XL** are loaded in half-point precision (FP16) whenever possible. This allows for faster inference and memory savings, since using them at full precision makes no difference to most people. When using Stable Diffusion XL models, it is possible to use under 10 VRAM, although there may be some specific cases with VAE decoding where it goes over this limit (which is still being investigated).

It is highly recommended to use the included VAE with the FP16 fix, since the VAE in the base model and most of the shared ones need to be used in full precision to avoid generating black images.

## Features

- Run Stable Diffusion XL models to generate images.
- User-friendly interface for easy image generation.
- Will completely run offline after the first installation.
- Powerfull features only avalaible as a desktop application.
- Easy sharing of models and Lora's metadata since the information its stored in each model, including sample image, sample generation, triggers and tags for filtering.

## Limitations

- Only runs with Stable Diffusion XL models.
- It has the default 75 CLIP token limitation for the prompts.

You can read why [here](https://github.com/ZCode-opensource/image-artisan-xl/blob/main/EXPLANATIONS.MD).

## Hardware requirements

- NVidia GPU with support for Cuda 11.8.
- 16 GB RAM for a 12 VRAM or more GPU.
- 24 GB RAM for lower than 12 VRAM cards (cpu offloading).
- 4 GB Minimum VRAM with sequential cpu offloading (takes more than 4x times to generate though).

## Planned features

- Image to image.
- Inpainting.
- Outpainting.
- ControlNet, T2I-Adapter and IP-Adapter.
- Upscaling.
- Fine tune your own model.
- Train LoRAs (diffusers and Kohya).
- Dataset creation and management.
- Gallery management.
- Nodes.

## Installation

### Windows installer

Please download the most recent installer from the [releases](https://github.com/ZCode-opensource/image-artisan-xl/releases) section. This version is exclusively for x64 and, while it has only been tested on Windows 11, it should also be compatible with Windows 10.

Before proceeding with the installation, ensure that you have the following prerequisites installed:

- [Git for windows](https://git-scm.com/download/win)
- [Powershell 7](https://learn.microsoft.com/en-us/powershell/)

The installer will open a separate window to download the required dependencies, this will take some time specially for pytorch which is a 2.8 GB install.

The final installed application without models will take around **6 GB** of space, it will be installed under a directory named ZCode/ImageArtisanXL and you can uninstall later from the `Windows installed apps menu`. It will create a shortcut icon on your desktop named "Image Artisan XL".

The application will have a self contained version of python and environment, it will also have the tiny decoder and fixed VAE for half precision.

Once installed, on the first run you will be asked to choose the directories for the models, you can use the defaults or point them to a place you already have with models. Later you can change them on the fly if you want to use separate model directories.

### Documentation

Coming soon...

### Manual install

#### Linux

```bash
python -m venv .venv --prompt ImageArtisanXL
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install .
```

Then run with the command `python -m iartisan`.

#### Windows

```powershell
python -m venv .venv --prompt ImageArtisanXL
.\.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install .
```

Then run with the command `python -m iartisanxl`.

#### Developer

Use this instead `pip install -e .[dev]` to install.

### Additional steps for manual installation

#### intermediate images

If you want to display the images when generating you'll need to create this directory structure inside the project:

> /models  
> -- /taesd

then download the taesd decoder from here:

https://github.com/madebyollin/taesd/blob/main/taesdxl_decoder.pth

and save it in this location.

#### Vae with half precision fixed

For using less VRAM when decoding the images you'll need to create this directory structure inside the project:

> /models  
> -- /vae-fp16

then download the vae model files from here:

https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/diffusion_pytorch_model.safetensors  
https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/blob/main/config.json

and save them in this location.

## License

Image Artisan XL is licensed under the MIT License. You can find the full text of the license [here](https://github.com/ZCode-opensource/image-artisan-xl/blob/main/LICENSE).

## Acknowledgements

Special thanks to:

[HarroweD](https://civitai.com/user/HarroweD/models) for his LoRA [Harrlogos XL](https://civitai.com/models/176555/harrlogos-xl-finally-custom-text-generation-in-sd) which was used to make the splashscreen.  
[artificialguybr](https://civitai.com/user/artificialguybr/models) for his LoRA [Icons.Redmond](https://civitai.com/models/122827/iconsredmond-app-icons-lora-for-sd-xl-10) which was used to make the app icon.

The creators of [InvokeAI](https://github.com/invoke-ai/InvokeAI), [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and [sd-scripts](https://github.com/kohya-ss/sd-scripts) used for inspiration.

All the contributtors to this projects:
[taesd](https://github.com/madebyollin/taesd),
[Diffusers](https://github.com/huggingface/diffusers),
[Transformers](https://github.com/huggingface/transformers),
[Peft](https://github.com/huggingface/peft)
and of course all the other big libraries that made this project possible.

Most vectors and icons by [SVG Repo](https://www.svgrepo.com)
