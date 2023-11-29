import unittest

from iartisanxl.generation.lora_data_object import LoraDataObject


class TestLoraDataObject(unittest.TestCase):
    def setUp(self):
        self.lora_data_object = LoraDataObject(
            name="test_name",
            filename="test_filename",
            version="test_version",
            path="test_path",
            enabled=True,
            weight=1.0,
            id=None,
        )

    def test_attributes(self):
        self.assertEqual(self.lora_data_object.name, "test_name")
        self.assertEqual(self.lora_data_object.filename, "test_filename")
        self.assertEqual(self.lora_data_object.version, "test_version")
        self.assertEqual(self.lora_data_object.path, "test_path")
        self.assertEqual(self.lora_data_object.enabled, True)
        self.assertEqual(self.lora_data_object.weight, 1.0)
        self.assertEqual(self.lora_data_object.id, None)
