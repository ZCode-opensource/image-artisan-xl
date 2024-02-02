import torch
import noise
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from iartisanxl.modules.common.image.image_utils import normalize_tensor_image


def mandelbrot(c, max_iter):
    z = c.clone()
    n = torch.zeros(c.shape).to(c.device)
    mask = torch.ones(c.shape).to(c.device).bool()
    for i in range(max_iter):
        z[mask] = z[mask] ** 2 + c[mask]
        mask = torch.abs(z) <= 1000
        n[mask] = i
    return n


def draw_mandelbrot(xmin, xmax, ymin, ymax, width, height, max_iter):
    x = torch.linspace(xmin, xmax, width)
    y = torch.linspace(ymin, ymax, height)
    c = x[None, :] + 1j * y[:, None]
    return mandelbrot(c, max_iter)


def create_mandelbrot_tensor(noise_scale, width, heigh):
    iterations = int(noise_scale * 1000)
    mandelbrot_tensor = draw_mandelbrot(-2.0, 1.0, -1.5, 1.5, width, heigh, iterations)
    im_array = mandelbrot_tensor.cpu().numpy()
    im_array = np.interp(im_array, (im_array.min(), im_array.max()), (0, 255)).astype(np.uint8)
    im = Image.fromarray(im_array)
    im = im.convert("L")
    colors = plt.get_cmap("gray")(im_array)

    mandelbrot_image = torch.from_numpy((colors[:, :, :3] * 255).astype(np.uint8)).permute(2, 0, 1).unsqueeze(0)
    mandelbrot_image = normalize_tensor_image(mandelbrot_image)

    return mandelbrot_image


def create_noise_tensor(noise_type, noise_scale, width, height):
    noise_image = np.zeros((width, height))
    noise_scale *= 1000
    noise_func = noise.pnoise2 if noise_type == "perlin" else noise.snoise2

    for i in range(width):
        for j in range(height):
            noise_image[i][j] = noise_func(
                i / noise_scale, j / noise_scale, octaves=6, persistence=0.5, lacunarity=2.0, repeatx=width, repeaty=height, base=42
            )

    noise_image = np.interp(noise_image, (noise_image.min(), noise_image.max()), (0, 255))
    noise_tensor = torch.from_numpy(noise_image).float().unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)

    return noise_tensor


def add_torch_noise(image, noise_type, noise_factor):
    if isinstance(image, Image.Image):
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div(255)

    if noise_type == "uniform":
        generated_noise = torch.rand(*image.shape, device=image.device) * noise_factor
    else:
        generated_noise = torch.randn(*image.shape, device=image.device) * noise_factor

    noisy_image = image + generated_noise
    noisy_image = torch.clamp(noisy_image, 0, 1)

    return noisy_image
