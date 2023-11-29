import unittest

from iartisanxl.generation.vae_data_object import VaeDataObject


class TestVaeDataObject(unittest.TestCase):
    def test_init(self):
        obj = VaeDataObject(name="test", path="/path/to/test")
        self.assertEqual(obj.name, "test")
        self.assertEqual(obj.path, "/path/to/test")
