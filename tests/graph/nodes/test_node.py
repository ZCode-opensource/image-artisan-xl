import unittest
from collections import defaultdict

from iartisanxl.graph.nodes.node import Node


class MockNode(Node):
    REQUIRED_INPUTS = ["input_1"]
    OUTPUTS = ["output_1"]


class TestNode(unittest.TestCase):
    def setUp(self):
        self.node = MockNode()

    def test_init(self):
        self.assertIsNone(self.node.id)
        self.assertTrue(self.node.updated)
        self.assertEqual(self.node.dependencies, [])
        self.assertEqual(self.node.dependents, [])
        self.assertEqual(self.node.values, {})
        self.assertIsInstance(self.node.connections, defaultdict)
        self.assertIsNone(self.node.device)
        self.assertIsNone(self.node.torch_dtype)
        self.assertFalse(self.node.cpu_offload)
        self.assertFalse(self.node.sequential_offload)

    def test_connect(self):
        node2 = MockNode()
        self.node.connect("input_1", node2, "output_1")
        self.assertIn(node2, self.node.dependencies)
        self.assertIn(self.node, node2.dependents)
        self.assertTrue(self.node.updated)

    def test_disconnect(self):
        node2 = MockNode()
        self.node.connect("input_1", node2, "output_1")
        self.node.disconnect("input_1", node2, "output_1")
        self.assertNotIn(node2, self.node.dependencies)
        self.assertNotIn(self.node, node2.dependents)
        self.assertTrue(self.node.updated)

    def test_connect_with_non_node(self):
        with self.assertRaises(TypeError):
            self.node.connect("input_1", "not a node", "output_1")

    def test_connect_with_invalid_output(self):
        node2 = MockNode()
        with self.assertRaises(ValueError):
            self.node.connect("input_1", node2, "invalid_output")

    def test_disconnect_from_node(self):
        node2 = MockNode()
        self.node.connect("input_1", node2, "output_1")
        self.node.disconnect_from_node(node2)
        self.assertNotIn(node2, self.node.dependencies)
        self.assertNotIn(self.node, node2.dependents)
        self.assertTrue(self.node.updated)
        for _input_name, conns in self.node.connections.items():
            self.assertNotIn(node2, [n for n, _output_name in conns])

    def test_getattr_with_invalid_attribute(self):
        with self.assertRaises(AttributeError):
            _ = self.node.invalid_attribute
