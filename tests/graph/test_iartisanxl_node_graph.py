import os
import tempfile
import json

import unittest
from unittest.mock import patch
from iartisanxl.graph.iartisanxl_node_graph import ImageArtisanNodeGraph
from iartisanxl.graph.nodes.node import Node


class MockNode(Node):
    pass


class MockNumberNode(Node):
    OUTPUTS = ["value"]

    def __init__(self, number_value=None):
        super().__init__()
        self.number = number_value

    def update_number(self, new_number_value):
        self.number = new_number_value
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["number"] = self.number
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(MockNumberNode, cls).from_dict(node_dict)
        node.number = node_dict["number"]
        return node

    def update_inputs(self, node_dict):
        self.number = node_dict["number"]
        self.set_updated()

    def __call__(self):
        super().__call__()
        self.values["value"] = self.number
        return self.values


class MockSumNumbers(Node):
    REQUIRED_INPUTS = ["number_one", "number_two"]
    OPTIONAL_INPUTS = ["number_three"]
    OUTPUTS = ["sum_numbers"]

    def __call__(self):
        super().__call__()
        sum_numbers = self.number_one + self.number_two
        if self.number_three is not None:
            sum_numbers += self.number_three
        self.values["sum_numbers"] = sum_numbers
        return self.values


class TestImageArtisanNodeGraph(unittest.TestCase):
    def setUp(self):
        self.graph = ImageArtisanNodeGraph()

    def test_add_node(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node)
        self.assertEqual(self.graph.nodes[0], mock_node)
        self.assertEqual(mock_node.id, 0)

    def test_get_node(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node)
        retrieved_node = self.graph.get_node(MockNode, 0)
        self.assertEqual(retrieved_node, mock_node)

    def test_delete_node(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node)
        self.graph.delete_node(MockNode, 0)
        self.assertEqual(len(self.graph.nodes), 0)

    @patch.object(MockNode, "__call__", return_value=None)
    def test_call(self, mock_call):
        mock_node = MockNode()
        self.graph.add_node(mock_node)
        self.graph()
        mock_call.assert_called_once()

    def test_node_required_interaction(self):
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=4)
        self.graph.add_node(mock_number_node2)

        mock_sum_node = MockSumNumbers()
        mock_sum_node.connect("number_one", mock_number_node1, "value")
        mock_sum_node.connect("number_two", mock_number_node2, "value")
        self.graph.add_node(mock_sum_node)

        self.graph()

        self.assertEqual(mock_sum_node.values["sum_numbers"], 7)

    def test_node_optional_interaction(self):
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=4)
        self.graph.add_node(mock_number_node2)

        mock_number_node3 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node3)

        mock_sum_node = MockSumNumbers()
        mock_sum_node.connect("number_one", mock_number_node1, "value")
        mock_sum_node.connect("number_two", mock_number_node2, "value")
        mock_sum_node.connect("number_three", mock_number_node3, "value")
        self.graph.add_node(mock_sum_node)

        self.graph()

        self.assertEqual(mock_sum_node.values["sum_numbers"], 10)

    def test_node_required_check(self):
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_sum_node = MockSumNumbers()
        mock_sum_node.connect("number_one", mock_number_node1, "value")
        self.graph.add_node(mock_sum_node)

        with self.assertRaises(ValueError) as context:
            self.graph()

        self.assertTrue(
            'The required input "number_two" is not connected in "MockSumNumbers"'
            in str(context.exception)
        )

    def test_save_and_load(self):
        # Add nodes to the graph
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=4)
        self.graph.add_node(mock_number_node2)

        mock_sum_node = MockSumNumbers()
        mock_sum_node.connect("number_one", mock_number_node1, "value")
        mock_sum_node.connect("number_two", mock_number_node2, "value")
        self.graph.add_node(mock_sum_node)

        # Save the graph to a temporary JSON file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.graph.save_to_json(temp_file.name)
        temp_file.close()  # Close the file

        # Load the JSON file into a new graph
        new_graph = ImageArtisanNodeGraph()
        new_graph.load_from_json(
            temp_file.name,
            {"MockNumberNode": MockNumberNode, "MockSumNumbers": MockSumNumbers},
        )

        # Check that the new graph is the same as the original one
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        for node in new_graph.nodes:
            self.assertIsInstance(node, (MockNumberNode, MockSumNumbers))

        # Clean up the temporary file)
        os.unlink(temp_file.name)

    def test_update_from_json(self):
        # Create a graph
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=5)
        self.graph.add_node(mock_number_node2)

        mock_sum_node = MockSumNumbers()
        mock_sum_node.connect("number_one", mock_number_node1, "value")
        mock_sum_node.connect("number_two", mock_number_node2, "value")
        self.graph.add_node(mock_sum_node)

        # run the graph
        self.graph()

        # test the initial sum
        self.assertEqual(mock_sum_node.values["sum_numbers"], 8)

        json_graph = {
            "nodes": [
                {"class": "MockNumberNode", "id": 0, "number": 5},
                {"class": "MockNumberNode", "id": 1, "number": 5},
                {"class": "MockSumNumbers", "id": 2},
            ],
            "connections": [
                {
                    "from_node_id": 0,
                    "from_output_name": "value",
                    "to_node_id": 2,
                    "to_input_name": "number_one",
                },
                {
                    "from_node_id": 1,
                    "from_output_name": "value",
                    "to_node_id": 2,
                    "to_input_name": "number_two",
                },
            ],
        }
        NODE_CLASSES = {
            "MockNumberNode": MockNumberNode,
            "MockSumNumbers": MockSumNumbers,
        }
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(temp_file.name, "w", encoding="utf-8") as f:
            json.dump(json_graph, f)
        temp_file.close()

        # Load the JSON file into a graph
        self.graph.update_from_json(temp_file.name, NODE_CLASSES)
        self.graph()

        # Check that the graph contains the expected node
        self.assertEqual(len(self.graph.nodes), 3)
        self.assertIsInstance(self.graph.nodes[0], MockNumberNode)
        self.assertIsInstance(self.graph.nodes[1], MockNumberNode)
        self.assertIsInstance(self.graph.nodes[2], MockSumNumbers)
        self.assertEqual(mock_sum_node.values["sum_numbers"], 10)

        # Clean up the temporary file
        os.unlink(temp_file.name)

    def test_update_from_json_without_changes(self):
        # Create a graph
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=5)
        self.graph.add_node(mock_number_node2)

        mock_sum_node = MockSumNumbers()
        mock_sum_node.connect("number_one", mock_number_node1, "value")
        mock_sum_node.connect("number_two", mock_number_node2, "value")
        self.graph.add_node(mock_sum_node)

        # run the graph
        self.graph()

        json_graph = {
            "nodes": [
                {"class": "MockNumberNode", "id": 0, "number": 3},
                {"class": "MockNumberNode", "id": 1, "number": 5},
                {"class": "MockSumNumbers", "id": 2},
            ],
            "connections": [
                {
                    "from_node_id": 0,
                    "from_output_name": "value",
                    "to_node_id": 2,
                    "to_input_name": "number_one",
                },
                {
                    "from_node_id": 1,
                    "from_output_name": "value",
                    "to_node_id": 2,
                    "to_input_name": "number_two",
                },
            ],
        }
        NODE_CLASSES = {
            "MockNumberNode": MockNumberNode,
            "MockSumNumbers": MockSumNumbers,
        }
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(temp_file.name, "w", encoding="utf-8") as f:
            json.dump(json_graph, f)
        temp_file.close()

        # Load the JSON file into a graph
        self.graph.update_from_json(temp_file.name, NODE_CLASSES)

        # Check the updated state of each node
        self.assertEqual(mock_number_node1.updated, False)
        self.assertEqual(mock_number_node2.updated, False)
        self.assertEqual(mock_sum_node.updated, False)

        # Clean up the temporary file
        os.unlink(temp_file.name)