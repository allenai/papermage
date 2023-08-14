"""

A (start, end) interval that references some string. For example, span (0, 4)
represents the word 'This' in the string 'This is a document.'

@kylel, @egork

"""

from typing import List, Dict, List

from collections import defaultdict


class Span:
    __slots__ = ['start', 'end']

    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def to_json(self) -> List[int]:
        """Returns whatever representation is JSON compatible"""
        return [self.start, self.end]

    @classmethod
    def from_json(cls, span_json: List) -> "Span":
        """Recreates the object from the JSON serialization"""
        return Span(start=span_json[0], end=span_json[-1])

    def __repr__(self):
        return f'Span{self.to_json()}'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Span):
            return False
        return self.start == other.start and self.end == other.end

    def __lt__(self, other: 'Span'):
        """Useful for sort(). Orders according to the start index.
        If ties, then order according to the end index."""
        if self.start == other.start:
            return self.end < other.end
        return self.start < other.start

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def is_overlap(self, other: 'Span') -> bool:
        """Whether self overlaps with the other Span object."""
        return (
            self.start <= other.start < self.end
            or other.start <= self.start < other.end
            or self == other
        )

    @classmethod
    def create_enclosing_span(cls, spans: List['Span']) -> 'Span':
        """Create the narrowest Span that completely encloses all the input Spans."""
        if not spans:
            raise ValueError(f'`spans` should be non-empty.')
        start = spans[0].start
        end = spans[0].end
        for span in spans[1:]:
            if span.start < start:
                start = span.start
            if span.end > end:
                end = span.end
        return Span(start=start, end=end)


class MergeClusterSpans:
    """
    Merge neighboring spans which are index distance apart
    Inspired by https://leetcode.com/problems/merge-intervals/

    Originally @egork, Revised @kylel
    """

    def __init__(
            self,
            spans: List[Span],
            index_distance: int = 1
    ) -> None:
        """
        Args
            index_distance (int): Distance between the spans
        """
        self._spans = spans
        self._index_distance = index_distance
        self._graph = self._build_graph(spans=spans, index_distance=index_distance)
        self._clusters = None

    @property
    def spans(self) -> List[Span]:
        return self._spans

    @property
    def index_distance(self) -> int:
        return self._index_distance

    @index_distance.setter
    def index_distance(self, d: int):
        """If modify this distance, everything that's been computed before
        should be recomputed."""
        self._index_distance = d
        self._graph = self._build_graph(spans=self.spans, index_distance=d)
        if self._clusters:
            self._clusters = self._cluster(spans=self.spans)

    @property
    def clusters(self) -> List[List[Span]]:
        if not self._clusters:
            self._clusters = self._cluster(spans=self.spans)
        return self._clusters

    @staticmethod
    def _is_neighboring_spans(span1: Span, span2: Span, index_distance: int) -> bool:
        """Whether two spans are considered neighboring"""
        return min(
            abs(span1.start - span2.end), abs(span1.end - span2.start)
        ) <= index_distance

    def _build_graph(self, spans: List[Span], index_distance: int) -> Dict[int, List[int]]:
        """
        Build graph, each node is the position within the input list of spans.
        Spans are considered overlapping if they are index_distance apart
        """
        graph = defaultdict(list)
        for i, span_i in enumerate(spans):
            for j in range(i + 1, len(spans)):
                if self._is_neighboring_spans(span1=span_i, span2=spans[j], index_distance=index_distance):
                    graph[i].append(j)
                    graph[j].append(i)
        return graph

    def _cluster(self, spans: List[Span]) -> List[List[Span]]:
        """Cluster nodes (i.e. spans) by finding connected components"""
        if len(spans) == 0:
            return [[]]

        visited = set()
        num_components = 0
        component_id_to_members = defaultdict(list)

        def _dfs(start: int):
            stack = [start]
            while stack:
                pos = stack.pop()
                if pos not in visited:
                    visited.add(pos)
                    component_id_to_members[num_components].append(pos)
                    stack.extend(self._graph[pos])

        # mark all nodes in the same connected component with the same integer.
        for i, span in enumerate(spans):
            if i not in visited:
                _dfs(start=i)
                num_components += 1

        return [
            [spans[member_id] for member_id in sorted(component_id_to_members[n])]
            for n in range(num_components)
        ]

    def merge(self) -> List[Span]:
        """
        For each of the lists of the connected nodes, merge into bigger Spans
        """
        merged_spans = []
        for cluster in self.clusters:
            if cluster:
                merged_span = Span.create_enclosing_span(spans=cluster)
                merged_spans.append(merged_span)
        return merged_spans
