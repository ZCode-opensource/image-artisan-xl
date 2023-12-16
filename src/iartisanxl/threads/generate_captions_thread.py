import io

from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal, QBuffer
from transformers import BlipProcessor, BlipForConditionalGeneration


class GenerateCaptionsThread(QThread):
    status_update = pyqtSignal(str)
    caption_done = pyqtSignal(str)

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.pixmap = None
        self.text = None
        self.processor = None
        self.model = None

    def run(self):
        if self.model is None:
            self.status_update.emit("Loading FuseCap model...")
            self.processor = BlipProcessor.from_pretrained("models/captions/fusecap")
            self.model = BlipForConditionalGeneration.from_pretrained("models/captions/fusecap").to(self.device)
            self.status_update.emit("FuseCap loaded.")

        self.status_update.emit("Generating AI caption...")

        qimage = self.pixmap.toImage()
        buffer = QBuffer()
        buffer.open(QBuffer.ReadWrite)
        qimage.save(buffer, "PNG")

        print(f"{self.text=}")

        raw_image = Image.open(io.BytesIO(buffer.data()))
        inputs = self.processor(raw_image, self.text, return_tensors="pt").to(self.device)

        outputs = self.model.generate(**inputs, num_beams=3)
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

        self.caption_done.emit(generated_text)
