import unittest

from iartisanxl.modules.common.lora.lora_list import LoraList
from iartisanxl.modules.common.lora.lora_data_object import LoraDataObject


class TestLoraList(unittest.TestCase):
    def setUp(self):
        self.lora_list = LoraList()
        self.lora_one = LoraDataObject(
            name="test_name",
            filename="test_filename",
            version="0.1",
            path="/path/to/lora_one",
            enabled=True,
            weight=0.5,
        )
        self.lora_two = LoraDataObject(
            name="test_name_two",
            filename="test_filename_two",
            version="0.8",
            path="/path/to/lora_two",
            enabled=True,
            weight=1.2,
        )

    def test_add(self):
        self.lora_list.add(self.lora_one)
        self.assertEqual(self.lora_list.loras[0], self.lora_one)

    def test_update_lora(self):
        self.lora_list.add(self.lora_one)
        self.lora_list.update_lora("test_filename", {"enabled": False})
        self.assertEqual(self.lora_list.loras[0].enabled, False)

    def test_get_lora_by_filename(self):
        self.lora_list.add(self.lora_one)
        self.assertEqual(self.lora_list.get_lora_by_filename("test_filename"), self.lora_one)

    def test_get_lora_by_bad_filename(self):
        self.lora_list.add(self.lora_one)
        self.assertEqual(self.lora_list.get_lora_by_filename("mock_filename"), None)

    def test_update_lora_by_id(self):
        lora_id = self.lora_list.add(self.lora_two)
        self.lora_list.update_lora_by_id(lora_id, {"name": "mock_name"})
        self.assertEqual(self.lora_list.loras[0].name, "mock_name")

    def test_update_filename_lora_by_id(self):
        self.lora_list.add(self.lora_two)
        self.lora_list.update_lora_by_id(2, {"filename": "mock_filename"})
        self.assertNotEqual(self.lora_list.loras[0].filename, "mock_filename")

    def test_get_lora_by_id(self):
        lora_id = self.lora_list.add(self.lora_two)
        self.assertEqual(self.lora_list.get_lora_by_id(lora_id), self.lora_two)

    def test_get_lora_by_bad_id(self):
        self.lora_list.add(self.lora_one)
        self.assertEqual(self.lora_list.get_lora_by_id(1), None)

    def test_remove(self):
        self.lora_list.add(self.lora_one)
        self.lora_list.add(self.lora_two)
        self.lora_list.remove(self.lora_one)
        self.assertEqual(len(self.lora_list.loras), 1)

    def test_save_state(self):
        self.lora_list.add(self.lora_one)
        self.lora_list.add(self.lora_two)
        self.lora_list.save_state()
        # pylint: disable=protected-access
        self.assertEqual(self.lora_list._original_loras[0], self.lora_one)
        self.assertEqual(self.lora_list._original_loras[1], self.lora_two)

    def test_get_added(self):
        self.lora_list.add(self.lora_one)
        self.lora_list.save_state()
        self.lora_list.add(self.lora_two)
        self.assertEqual(self.lora_list.get_added()[0], self.lora_two)

    def test_get_removed(self):
        self.lora_list.add(self.lora_one)
        self.lora_list.add(self.lora_two)
        self.lora_list.save_state()
        self.lora_list.remove(self.lora_one)
        self.assertEqual(self.lora_list.get_removed()[0], self.lora_one)

    def test_get_modified(self):
        self.lora_list.add(self.lora_one)
        self.lora_list.add(self.lora_two)
        self.lora_list.save_state()
        self.lora_list.update_lora("test_filename", {"enabled": False})
        self.assertEqual(len(self.lora_list.get_modified()), 1)
        self.assertEqual(self.lora_list.get_modified()[0], self.lora_one)
        self.assertEqual(len(self.lora_list.get_added()), 0)
