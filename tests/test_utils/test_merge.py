"""

@kylel

"""


import unittest

from papermage.magelib import Span
from papermage.utils.merge import ClusterMergeResults, cluster_and_merge_neighbor_spans


class TestClusterMergeResults(unittest.TestCase):
    def test_num_clusters(self):
        result = ClusterMergeResults(
            items=["a", "b", "c"], cluster_ids=[[0, 1], [2]], cluster_id_to_merged={0: "ab", 1: "c"}
        )
        self.assertEqual(result.num_clusters, 2)

    def test_clusters(self):
        result = ClusterMergeResults(
            items=["a", "b", "c"], cluster_ids=[[0, 1], [2]], cluster_id_to_merged={0: "ab", 1: "c"}
        )
        self.assertListEqual(result.clusters, [["a", "b"], ["c"]])

    def test_merged(self):
        result = ClusterMergeResults(
            items=["a", "b", "c"], cluster_ids=[[0, 1], [2]], cluster_id_to_merged={0: "ab", 1: "c"}
        )
        self.assertListEqual(result.merged, ["ab", "c"])

    def test_get_merged(self):
        result = ClusterMergeResults(
            items=["a", "b", "c"], cluster_ids=[[0, 1], [2]], cluster_id_to_merged={0: "ab", 1: "c"}
        )
        self.assertEqual(result.get_merged("a"), "ab")
        self.assertEqual(result.get_merged("b"), "ab")
        self.assertEqual(result.get_merged("c"), "c")


class TestMergeClusterSpans(unittest.TestCase):
    def setUp(self):
        self.a = Span(0, 1)
        self.b = Span(1, 2)
        self.c = Span(3, 4)
        self.d = Span(4, 5)
        self.e = Span(9, 10)

    def test_empty(self):
        out = cluster_and_merge_neighbor_spans(spans=[])
        self.assertEqual(out.num_clusters, 0)
        self.assertListEqual(out.clusters, [])
        self.assertListEqual(out.merged, [])

    def test_single(self):
        out = cluster_and_merge_neighbor_spans(spans=[self.a])
        self.assertEqual(out.num_clusters, 1)
        self.assertListEqual(out.clusters, [[self.a]])
        self.assertListEqual(out.merged, [self.a])

    def test_disjoint(self):
        out = cluster_and_merge_neighbor_spans(spans=[self.a, self.c, self.e])
        self.assertEqual(out.num_clusters, 3)
        self.assertListEqual(out.clusters, [[self.a], [self.c], [self.e]])
        self.assertListEqual(out.merged, [self.a, self.c, self.e])
        # reversed order
        out = cluster_and_merge_neighbor_spans(spans=[self.e, self.c, self.a])
        self.assertEqual(out.num_clusters, 3)
        self.assertListEqual(out.clusters, [[self.a], [self.c], [self.e]])
        self.assertListEqual(out.merged, [self.a, self.c, self.e])

    def test_cluster(self):
        out = cluster_and_merge_neighbor_spans(spans=[self.a, self.b, self.c, self.d, self.e], distance=0)
        self.assertEqual(out.num_clusters, 3)
        self.assertListEqual(out.clusters, [[self.a, self.b], [self.c, self.d], [self.e]])
        self.assertListEqual(out.merged, [Span(0, 2), Span(3, 5), Span(9, 10)])
        out = cluster_and_merge_neighbor_spans(spans=[self.a, self.b, self.c, self.d, self.e], distance=1)
        self.assertEqual(out.num_clusters, 2)
        self.assertListEqual(out.clusters, [[self.a, self.b, self.c, self.d], [self.e]])
        self.assertListEqual(out.merged, [Span(0, 5), Span(9, 10)])
        out = cluster_and_merge_neighbor_spans(spans=[self.a, self.b, self.c, self.d, self.e], distance=4)
        self.assertEqual(out.num_clusters, 1)
        self.assertListEqual(out.clusters, [[self.a, self.b, self.c, self.d, self.e]])
        self.assertListEqual(out.merged, [Span(0, 10)])
