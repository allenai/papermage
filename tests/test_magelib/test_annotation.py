"""

@kylel

"""

import unittest

from papermage.magelib import Annotation


class DummyAnnotation(Annotation):
    pass


class DummyDoc:
    pass


class TestAnnotation(unittest.TestCase):
    def test_abstract_methods(self):
        # trick to test abstract methods from
        # https://clamytoe.github.io/articles/2020/Mar/12/testing-abcs-with-abstract-methods-with-pytest/
        Annotation.__abstractmethods__ = set()
        a = DummyAnnotation()

        to_json = a.to_json()
        from_json = a.from_json(annotation_json={"doesnt_matter": "whats_here"})
        self.assertIsNone(to_json)
        self.assertIsNone(from_json)

    def test_doc(self):
        d = DummyDoc()
        a = Annotation()

        # defaults to None
        self.assertIsNone(a.doc)

        # attaches reference to the Doc object
        a.doc = d
        self.assertIs(a.doc, d)

        # protected setter
        with self.assertRaises(AttributeError) as e:
            a.doc = DummyDoc()

        # detaches from Doc
        a.doc = None
        self.assertIsNone(a.doc)

    def test_id(self):
        a = Annotation()

        # defaults to None
        self.assertIsNone(a.id)

        # setting id doesnt work without Doc
        with self.assertRaises(AttributeError):
            a.id = 12345

        # setting id works w/ a Doc first
        d = DummyDoc()
        a.doc = d
        a.id = 12345
        self.assertEqual(a.id, 12345)
