import unittest

from iartisanxl.generation.model_data_object import ModelDataObject


class TestModelDataObject(unittest.TestCase):
    def test_init(self):
        obj = ModelDataObject(
            name="test", path="/path/to/test", version="1.0", type="test_type"
        )
        self.assertEqual(obj.name, "test")
        self.assertEqual(obj.path, "/path/to/test")
        self.assertEqual(obj.version, "1.0")
        self.assertEqual(obj.type, "test_type")
