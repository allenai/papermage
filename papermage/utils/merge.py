"""

Utilty methods related to Merging Entities. 

@kylel, @lucas

"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from papermage.magelib import Span


class ClusterMergeResults:
    def __init__(
        self, items: List[Any], cluster_ids: List[List[int]], cluster_id_to_merged: Dict[int, Any]
    ) -> None:
        self.cluster_ids = cluster_ids
        self.items = items
        self.cluster_id_to_merged = cluster_id_to_merged
        self.item_id_to_cluster_id = {
            item_id: cluster_id for cluster_id, cluster in enumerate(cluster_ids) for item_id in cluster
        }

    def __repr__(self) -> str:
        return f"ClusterMergeResult(clusters={self.clusters}, cluster_id_to_merged={self.cluster_id_to_merged})"

    @property
    def num_clusters(self) -> int:
        return len(self.cluster_ids)

    @property
    def clusters(self) -> List[List[Any]]:
        """Instead of returning the indices of the items, return the items themselves."""
        if self.num_clusters:
            return [[self.items[item_id] for item_id in cluster] for cluster in self.cluster_ids]
        else:
            return []

    @property
    def merged(self) -> List[Any]:
        """Return the merged items."""
        if self.num_clusters:
            return [self.cluster_id_to_merged[cluster_id] for cluster_id in range(self.num_clusters)]
        else:
            return []

    def get_merged(self, item: Any) -> Any:
        """Return the merged item of the input item."""
        try:
            this_item_id = self.items.index(item)
        except ValueError:
            raise ValueError(f"Item {item} not found in the input items.")
        return self.cluster_id_to_merged[self.item_id_to_cluster_id[this_item_id]]


def cluster_and_merge_neighbor_spans(spans: List[Span], distance=1) -> ClusterMergeResults:
    """Merge neighboring spans in a list of un-overlapped spans:
    when the gaps between neighboring spans is smaller or equal to the
    specified distance, they are considered as the neighbors.

    Args:
        spans (List[Span]): The input list of spans.
        distance (int, optional):
            The upper bound of interval gaps between two neighboring spans.
            Defaults to 1.

    Returns:
        List[List[int]]: A list of clusters of neighboring spans. Each cluster is a list of indices of the spans.
        Dict[int, Span]: A dictionary mapping the cluster id to the merged span.
    """

    if len(spans) == 0:
        return ClusterMergeResults(items=[], cluster_ids=[], cluster_id_to_merged={})

    if len(spans) == 1:
        return ClusterMergeResults(items=spans, cluster_ids=[[0]], cluster_id_to_merged={0: spans[0]})

    # When sorted, only one iteration round is needed.
    spans = sorted(spans)

    is_neighboring_spans = (
        lambda span1, span2: min(abs(span1.start - span2.end), abs(span1.end - span2.start)) <= distance
    )

    # cluster and merge neighbors
    cur_cluster = 0
    cluster_ids = [[0]]
    cluster_id_to_big_span = {0: spans[0]}
    for span_id in range(1, len(spans)):
        cur_span = spans[span_id]
        cur_big_span = cluster_id_to_big_span[cur_cluster]
        if is_neighboring_spans(cur_span, cur_big_span):
            cluster_ids[cur_cluster].append(span_id)
            new_big_span = Span.create_enclosing_span(spans=[cur_big_span, cur_span])
            cluster_id_to_big_span[cur_cluster] = new_big_span
        else:
            cur_cluster += 1
            cluster_ids.append([span_id])
            cluster_id_to_big_span[cur_cluster] = cur_span

    return ClusterMergeResults(items=spans, cluster_ids=cluster_ids, cluster_id_to_merged=cluster_id_to_big_span)
