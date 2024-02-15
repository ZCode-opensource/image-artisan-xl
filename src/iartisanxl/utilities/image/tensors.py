import torch


OPENAI_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
OPENAI_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def normalize_tensor_image(image, mean=None, std=None):
    if mean is None:
        mean = OPENAI_CLIP_MEAN

    if std is None:
        std = OPENAI_CLIP_STD

    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return (image - mean) / std
