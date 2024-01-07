import os

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class LocalImageTextDataset(Dataset):
    def __init__(self, data_dir, tokenizer_one, tokenizer_two, size):
        self.data_dir = data_dir
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.size = size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.image_files = sorted([f for f in os.listdir(data_dir) if f.endswith((".jpg", ".jpeg", ".png"))])
        self.text_files = [os.path.splitext(f)[0] + ".txt" for f in self.image_files]

        self.buckets = {
            "1024x1024": [],
            "896x1152": [],
            "1152x896": [],
            "1344x768": [],
            "1536x704": [],
        }

        for i, image_file in enumerate(self.image_files):
            image = Image.open(os.path.join(self.data_dir, image_file))
            size = f"{image.width}x{image.height}"
            if size in self.buckets:
                self.buckets[size].append(i)

    def __getitem__(self, index):
        for _size, indices in self.buckets.items():
            if index < len(indices):
                image_file = self.image_files[indices[index]]
                break
            else:
                index -= len(indices)

        text_file = self.text_files[index]

        image = Image.open(os.path.join(self.data_dir, image_file))
        original_size = (image.height, image.width)

        train_resize = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR)
        train_crop = transforms.CenterCrop(self.size)

        image = train_resize(image)

        y1 = max(0, int(round((image.height - self.size) / 2.0)))
        x1 = max(0, int(round((image.width - self.size) / 2.0)))
        image = train_crop(image)
        crop_top_left = (y1, x1)
        image = self.transform(image)

        with open(os.path.join(self.data_dir, text_file), "r", encoding="utf-8") as f:
            text = f.read()

        tokens_one = self.tokenize_captions(text, self.tokenizer_one).squeeze(dim=0)
        tokens_two = self.tokenize_captions(text, self.tokenizer_two).squeeze(dim=0)

        return {
            "pixel_values": image,
            "input_ids_one": tokens_one,
            "input_ids_two": tokens_two,
            "original_size": original_size,
            "crop_top_left": crop_top_left,
        }

    def tokenize_captions(self, caption, tokenizer):
        text_inputs = tokenizer(
            caption,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return text_inputs.input_ids

    def __len__(self):
        return len(self.image_files)
