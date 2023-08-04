"""

Tests for Metadata

@lucas

"""

import unittest
from copy import deepcopy

from papermage.magelib import Metadata


class TestMetadata(unittest.TestCase):
    def test_add_keys(self):
        metadata = Metadata()

        metadata["foo"] = 1
        self.assertEqual(metadata.foo, 1)

        metadata.bar = 2
        self.assertEqual(metadata.bar, 2)

        metadata.set("baz", 3)
        self.assertEqual(metadata.baz, 3)

    def test_access_keys(self):
        metadata = Metadata()
        metadata.foo = "bar"

        self.assertEqual(metadata.foo, "bar")
        self.assertEqual(metadata.get("foo"), "bar")
        self.assertTrue(metadata["foo"])
        self.assertIsNone(metadata.get("bar"))

    def test_json_transform(self):
        metadata = Metadata.from_json({"foo": "bar"})

        self.assertEqual(metadata.to_json(), {"foo": "bar"})
        self.assertEqual(Metadata.from_json(metadata.to_json()), metadata)

    def test_len(self):
        metadata = Metadata.from_json({f"k{i}": i for i in range(10)})
        self.assertEqual(len(metadata), 10)

        metadata.pop("k0")
        self.assertEqual(len(metadata), 9)

        del metadata.k1
        self.assertEqual(len(metadata), 8)

    def test_valid_names(self):
        metadata = Metadata()

        # this should work fine
        metadata.set("foo", "bar")
        self.assertEqual(metadata.foo, "bar")

        # this should fail because `1foo` is not a valid python variable name
        with self.assertRaises(ValueError):
            metadata.set("1foo", "bar")

    def test_deep_copy(self):
        metadata = Metadata.from_json({"foo": 1, "bar": 2, "baz": 3})
        metadata2 = deepcopy(metadata)
        self.assertEqual(metadata, metadata2)

    def test_get_unknown_key(self):
        metadata = Metadata()
        self.assertIsNone(metadata.text)
