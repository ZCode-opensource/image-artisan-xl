import os
import tempfile
import json
import time

import unittest
from unittest.mock import patch
from iartisanxl.graph.iartisanxl_node_graph import ImageArtisanNodeGraph
from iartisanxl.graph.nodes.node import Node


class MockNode(Node):
    pass


class MockTextNode(Node):
    OUTPUTS = ["text"]

    def __init__(self, text: str = None):
        super().__init__()
        self.text = text

    def update_text(self, text: str):
        self.text = text
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["text"] = self.text
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(MockTextNode, cls).from_dict(node_dict)
        node.text = node_dict["text"]
        return node

    def update_inputs(self, node_dict):
        self.text = node_dict["text"]
        self.set_updated()

    def __call__(self):
        super().__call__()
        self.values["text"] = self.text
        return self.values


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


class MockMaybeSumNumbers(Node):
    REQUIRED_INPUTS = ["number_one"]
    OPTIONAL_INPUTS = ["number_two"]
    OUTPUTS = ["sum_numbers"]

    def __call__(self):
        super().__call__()
        sum_numbers = self.number_one
        if self.number_two is not None:
            sum_numbers += self.number_two
        self.values["sum_numbers"] = sum_numbers
        return self.values


class MockCycleNode(Node):
    REQUIRED_INPUTS = ["input_value"]
    OUTPUTS = ["output_value"]

    def __init__(self, value=None):
        super().__init__()
        self.value = value

    def __call__(self):
        super().__call__()
        self.values["output_value"] = self.value
        return self.values


class MockTimerNode(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self):
        super().__call__()
        time.sleep(0.1)


NODE_CLASSES = {
    "MockNode": MockNode,
    "MockTextNode": MockTextNode,
    "MockNumberNode": MockNumberNode,
    "MockSumNumbers": MockSumNumbers,
    "MockMaybeSumNumbers": MockMaybeSumNumbers,
    "MockCycleNode": MockCycleNode,
    "MockTimerNode": MockTimerNode,
}


class TestImageArtisanNodeGraph(unittest.TestCase):
    def setUp(self):
        self.graph = ImageArtisanNodeGraph()

    def test_add_node(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node)
        self.assertEqual(self.graph.nodes[0], mock_node)
        self.assertEqual(len(self.graph.nodes), 1)
        self.assertEqual(mock_node.id, 0)

    def test_get_node(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node)
        retrieved_node = self.graph.get_node(0)
        self.assertEqual(retrieved_node, mock_node)

    def test_get_false_node(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node)
        retrieved_node = self.graph.get_node(1)
        self.assertEqual(retrieved_node, None)

    def test_add_node_with_name(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node, "mock_node")
        self.assertEqual(self.graph.nodes[0].name, "mock_node")

    def test_add_duplicate_node_with_name(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node, "mock_node")

        with self.assertRaises(ValueError) as context:
            self.graph.add_node(mock_node, "mock_node")

        self.assertTrue(
            "A node with the name mock_node already exists in the graph."
            in str(context.exception)
        )

    def test_get_node_by_name(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node, "mock_node")
        retrieved_node = self.graph.get_node_by_name("mock_node")
        self.assertEqual(retrieved_node, mock_node)

    def test_get_false_node_by_name(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node, "mock_node")
        retrieved_node = self.graph.get_node_by_name("mock_false_node")
        self.assertEqual(retrieved_node, None)

    def test_delete_node(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node)
        self.graph.delete_node(0)
        self.assertEqual(len(self.graph.nodes), 0)

    def test_delete_node_by_name(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node, "mock_node")
        self.graph.delete_node_by_name("mock_node")
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

        # Save the graph to a variable
        json_graph = self.graph.to_json()

        # Load the JSON file into a new graph
        new_graph = ImageArtisanNodeGraph()
        new_graph.from_json(
            json_graph,
            NODE_CLASSES,
        )

        # Check that the new graph is the same as the original one
        self.assertEqual(len(new_graph.nodes), len(self.graph.nodes))
        for node in new_graph.nodes:
            self.assertIsInstance(node, (MockNumberNode, MockSumNumbers))

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 3)

    def test_save_and_load_with_file(self):
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

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 3)

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

        json_string = {
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

        json_graph = json.dumps(json_string)

        # Load the JSON into a graph
        self.graph.update_from_json(json_graph, NODE_CLASSES)

        # Check that the graph contains the expected nodes
        self.assertEqual(len(self.graph.nodes), 3)
        self.assertIsInstance(self.graph.nodes[0], MockNumberNode)
        self.assertIsInstance(self.graph.nodes[1], MockNumberNode)
        self.assertIsInstance(self.graph.nodes[2], MockSumNumbers)

        # Check updated state on each node
        self.assertEqual(mock_number_node1.updated, True)
        self.assertEqual(mock_number_node2.updated, False)
        self.assertEqual(mock_sum_node.updated, True)

        # run the graph again
        self.graph()

        # Check that the connections and the result of sum_node are correct
        self.assertEqual(mock_sum_node.values["sum_numbers"], 10)

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 3)

    def test_update_from_json_file(self):
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

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(temp_file.name, "w", encoding="utf-8") as f:
            json.dump(json_graph, f)
        temp_file.close()

        # Load the JSON file into a graph
        self.graph.update_from_json_file(temp_file.name, NODE_CLASSES)

        # Check that the graph contains the expected nodes
        self.assertEqual(len(self.graph.nodes), 3)
        self.assertIsInstance(self.graph.nodes[0], MockNumberNode)
        self.assertIsInstance(self.graph.nodes[1], MockNumberNode)
        self.assertIsInstance(self.graph.nodes[2], MockSumNumbers)

        # Check updated state on each node
        self.assertEqual(mock_number_node1.updated, True)
        self.assertEqual(mock_number_node2.updated, False)
        self.assertEqual(mock_sum_node.updated, True)

        # run the graph again
        self.graph()

        # Check that the connections and the result of sum_node are correct
        self.assertEqual(mock_sum_node.values["sum_numbers"], 10)

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 3)

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

        json_string = {
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

        json_graph = json.dumps(json_string)

        # Load the JSON into a graph
        self.graph.update_from_json(json_graph, NODE_CLASSES)

        # Check the updated state of each node
        self.assertEqual(mock_number_node1.updated, False)
        self.assertEqual(mock_number_node2.updated, False)
        self.assertEqual(mock_sum_node.updated, False)

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 3)

    def test_update_from_json_without_connections(self):
        # Create a graph
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=8)
        self.graph.add_node(mock_number_node2)

        # run the graph
        self.graph()

        json_string = {
            "nodes": [
                {"class": "MockNumberNode", "id": 0, "number": 3},
                {"class": "MockNumberNode", "id": 1, "number": 5},
            ],
            "connections": [],
        }

        json_graph = json.dumps(json_string)

        # Load the JSON into a graph
        self.graph.update_from_json(json_graph, NODE_CLASSES)

        # Check the updated state of each node
        self.assertEqual(mock_number_node1.updated, False)
        self.assertEqual(mock_number_node2.updated, True)

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 2)

    def test_update_from_json_adding_node(self):
        # Create a graph
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=8)
        self.graph.add_node(mock_number_node2)

        # run the graph
        self.graph()

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 2)

        json_string = {
            "nodes": [
                {"class": "MockNumberNode", "id": 0, "number": 3},
                {"class": "MockNumberNode", "id": 1, "number": 5},
                {"class": "MockSumNumbers", "id": 2},
            ],
            "connections": [],
        }

        json_graph = json.dumps(json_string)

        # Load the JSON into a graph
        self.graph.update_from_json(json_graph, NODE_CLASSES)

        self.assertEqual(len(self.graph.nodes), 3)
        self.assertIsInstance(self.graph.nodes[2], MockSumNumbers)

    def test_update_from_json_add_node_and_connections(self):
        # Create a graph
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=8)
        self.graph.add_node(mock_number_node2)

        # run the graph
        self.graph()

        json_string = {
            "nodes": [
                {"class": "MockNumberNode", "id": 0, "number": 3},
                {"class": "MockNumberNode", "id": 1, "number": 5},
                {"class": "MockNumberNode", "id": 2, "number": 10},
                {"class": "MockSumNumbers", "id": 3},
            ],
            "connections": [
                {
                    "from_node_id": 0,
                    "from_output_name": "value",
                    "to_node_id": 3,
                    "to_input_name": "number_one",
                },
                {
                    "from_node_id": 2,
                    "from_output_name": "value",
                    "to_node_id": 3,
                    "to_input_name": "number_two",
                },
            ],
        }

        json_graph = json.dumps(json_string)

        # Load the JSON into a graph
        self.graph.update_from_json(json_graph, NODE_CLASSES)

        self.assertEqual(len(self.graph.nodes), 4)
        self.assertIsInstance(self.graph.nodes[2], MockNumberNode)
        self.assertIsInstance(self.graph.nodes[3], MockSumNumbers)

        # run the graph
        self.graph()

        # Check that the connections and the result of sum_node are correct
        mock_sum_node = self.graph.nodes[3]
        self.assertEqual(mock_sum_node.values["sum_numbers"], 13)

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 4)

    def test_update_from_json_add_connections(self):
        # Create a graph
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=5)
        self.graph.add_node(mock_number_node2)

        mock_sum_node = MockMaybeSumNumbers()
        mock_sum_node.connect("number_one", mock_number_node1, "value")
        self.graph.add_node(mock_sum_node)

        # run the graph
        self.graph()

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 3)

        # Check value
        self.assertEqual(mock_sum_node.values["sum_numbers"], 3)

        json_string = {
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

        json_graph = json.dumps(json_string)

        # Load the JSON file into a graph
        self.graph.update_from_json(json_graph, NODE_CLASSES)

        # Check updated state on old nodes
        self.assertEqual(mock_number_node1.updated, False)
        self.assertEqual(mock_number_node2.updated, False)

        # Check updated on new node
        new_sum_node = self.graph.get_node(2)
        self.assertEqual(new_sum_node.updated, True)

        # run the graph again
        self.graph()

        # Check that the connections and the result of new sum_node are correct
        self.assertEqual(new_sum_node.values["sum_numbers"], 8)

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 3)

    def test_update_from_json_remove_node(self):
        # Create a graph
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=8)
        self.graph.add_node(mock_number_node2)

        # run the graph
        self.graph()

        json_string = {
            "nodes": [
                {"class": "MockNumberNode", "id": 0, "number": 3},
            ],
            "connections": [],
        }

        json_graph = json.dumps(json_string)

        # Load the JSON file into a graph
        self.graph.update_from_json(json_graph, NODE_CLASSES)

        # check number of nodes
        self.assertEqual(len(self.graph.nodes), 1)

        # Check the updated state of each node
        self.assertEqual(mock_number_node1.updated, False)

    def test_input_value_error(self):
        # Create a graph
        mock_number_node1 = MockNumberNode(number_value=3)
        self.graph.add_node(mock_number_node1)

        mock_number_node2 = MockNumberNode(number_value=8)
        self.graph.add_node(mock_number_node2)

        mock_sum_node = MockSumNumbers()
        mock_sum_node.connect("number_one", mock_number_node1, "value")
        mock_sum_node.connect("number_two", mock_number_node2, "value")
        self.graph.add_node(mock_sum_node)

        with self.assertRaises(ValueError) as context:
            mock_number_node1.connect("value", mock_sum_node, "sum_numbers")

        self.assertTrue(
            'The input "value" is not present in "MockNumberNode"'
            in str(context.exception)
        )

    def test_graph_cycle_error(self):
        # Create a graph
        mock_cycle_node1 = MockCycleNode(value=3)
        self.graph.add_node(mock_cycle_node1)

        mock_cycle_node2 = MockCycleNode()
        mock_cycle_node2.connect("input_value", mock_cycle_node1, "output_value")
        self.graph.add_node(mock_cycle_node2)

        # Create a cycle
        mock_cycle_node1.connect("input_value", mock_cycle_node2, "output_value")

        # Check if ValueError is raised when running the graph
        with self.assertRaises(ValueError):
            self.graph()

    def test_to_json_node_names(self):
        mock_node_one = MockNode()
        self.graph.add_node(mock_node_one, "test_node_one")
        mock_node_two = MockNode()
        self.graph.add_node(mock_node_two, "test_node_two")

        json_graph = self.graph.to_json()

        json_string = {
            "nodes": [
                {"class": "MockNode", "id": 0, "name": "test_node_one"},
                {"class": "MockNode", "id": 1, "name": "test_node_two"},
            ],
            "connections": [],
        }
        valid_json = json.dumps(json_string)

        self.assertEqual(json_graph, valid_json)

    def test_update_node_counter(self):
        self.assertEqual(self.graph.node_counter, 0)

        mock_node_one = MockNode()
        self.graph.add_node(mock_node_one)

        self.assertEqual(self.graph.node_counter, 1)

        mock_node_two = MockNode()
        self.graph.add_node(mock_node_two)

        self.assertEqual(self.graph.node_counter, 2)

    def test_update_node_counter_on_from_json(self):
        self.assertEqual(self.graph.node_counter, 0)

        json_string = {
            "nodes": [
                {"class": "MockNode", "id": 0},
                {"class": "MockNode", "id": 1},
                {"class": "MockNode", "id": 2},
            ],
            "connections": [],
        }

        json_graph = json.dumps(json_string)

        self.graph.from_json(json_graph, NODE_CLASSES)

        self.assertEqual(self.graph.node_counter, 3)

    def test_update_node_counter_random_on_from_json(self):
        self.assertEqual(self.graph.node_counter, 0)

        json_string = {
            "nodes": [
                {"class": "MockNode", "id": 0},
                {"class": "MockNode", "id": 11},
                {"class": "MockNode", "id": 3},
            ],
            "connections": [],
        }

        json_graph = json.dumps(json_string)

        self.graph.from_json(json_graph, NODE_CLASSES)

        self.assertEqual(self.graph.node_counter, 12)

    def test_update_node_counter_on_update_from_json(self):
        self.assertEqual(self.graph.node_counter, 0)

        mock_node_one = MockNode()
        self.graph.add_node(mock_node_one)

        self.assertEqual(self.graph.node_counter, 1)

        json_string = {
            "nodes": [
                {"class": "MockNode", "id": 1},
                {"class": "MockNode", "id": 4},
                {"class": "MockNode", "id": 3},
            ],
            "connections": [],
        }

        json_graph = json.dumps(json_string)

        self.graph.update_from_json(json_graph, NODE_CLASSES)

        self.assertEqual(self.graph.node_counter, 5)

    def test_node_deletion(self):
        # Add a node to the graph
        mock_node = MockNode()
        self.graph.add_node(mock_node)

        # Store the id of the node
        node_id = mock_node.id

        # Delete the node
        self.graph.delete_node(node_id)

        # Check that the node is not in the graph
        self.assertIsNone(self.graph.get_node(node_id))

        # Check that there are no references to the node in the graph
        for node in self.graph.nodes:
            self.assertNotIn(node_id, [dep.id for dep in node.dependencies])
            for conns in node.connections.values():
                self.assertNotIn(node_id, [n.id for n, _ in conns])

    def test_graph_updated_flag(self):
        # check that the graphs starts as not updated
        self.assertEqual(self.graph.updated, False)

        mock_number_node_one = MockNumberNode(number_value=12)
        self.graph.add_node(mock_number_node_one)

        mock_number_node_two = MockNumberNode(number_value=11)
        self.graph.add_node(mock_number_node_two)

        mock_text_node = MockTextNode(text="This text is a test")
        self.graph.add_node(mock_text_node)

        mock_sum_node = MockSumNumbers()
        mock_sum_node.connect("number_one", mock_number_node_one, "value")
        mock_sum_node.connect("number_two", mock_number_node_two, "value")
        self.graph.add_node(mock_sum_node)

        # Run the graph
        self.graph()

        # check if the graph was updated
        self.assertEqual(self.graph.updated, True)

        # First check the values of each node
        self.assertEqual(mock_number_node_one.number, 12)
        self.assertEqual(mock_number_node_two.number, 11)
        self.assertEqual(mock_text_node.text, "This text is a test")
        self.assertEqual(mock_sum_node.values["sum_numbers"], 23)

        # Run it again without updates
        self.graph()

        # check if the graph was updated
        self.assertEqual(self.graph.updated, False)

        # check that the sum did not change
        self.assertEqual(mock_sum_node.values["sum_numbers"], 23)

        # change manually the text node
        mock_text_node.update_text("This text is changed for this test")

        # check than only the text node was flagges as updated
        self.assertEqual(mock_number_node_one.updated, False)
        self.assertEqual(mock_number_node_two.updated, False)
        self.assertEqual(mock_text_node.updated, True)
        self.assertEqual(mock_sum_node.updated, False)

        # Run the graph
        self.graph()

        # check if the graph was updated
        self.assertEqual(self.graph.updated, True)

        # now lets test with from json
        json_string = {
            "nodes": [
                {"class": "MockNumberNode", "id": 0, "number": 12},
                {"class": "MockNumberNode", "id": 1, "number": 11},
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

        json_graph = json.dumps(json_string)
        self.graph.update_from_json(json_graph, NODE_CLASSES)

        self.assertEqual(mock_number_node_one.updated, False)
        self.assertEqual(mock_number_node_two.updated, False)

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 3)

        # check if the graph was updated
        self.assertEqual(self.graph.updated, True)

        # Now test if the graph is loaded from json
        self.graph.from_json(json_graph, NODE_CLASSES)

        # Test the number of nodes
        self.assertEqual(len(self.graph.nodes), 3)

    def test_node_elapsed_time(self):
        mock_node = MockNode()
        self.graph.add_node(mock_node)

        mock_timer_node = MockTimerNode()
        self.graph.add_node(mock_timer_node)

        # Run the graph
        self.graph()

        # Check that elapsed_time has been set for each node
        self.assertEqual(mock_node.elapsed_time, 0)
        self.assertGreater(mock_timer_node.elapsed_time, 0.1)
