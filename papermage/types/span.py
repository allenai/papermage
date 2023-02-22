"""

A (start, end) interval that references some string. For example, span (0, 4)
represents the word 'This' in the string 'This is a document.'

@kylel

"""

from typing import List, Optional, Dict, Tuple, List
from dataclasses import dataclass


@dataclass
class Span:
    start: int
    end: int

    def to_json(self) -> List[int]:
        """Returns whatever representation is JSON compatible"""
        return [self.start, self.end]

    @classmethod
    def from_json(cls, span_json: List) -> "Span":
        """Recreates the object from the JSON serialization"""
        return Span(start=span_json[0], end=span_json[-1])

    def __lt__(self, other: 'Span'):
        """Useful for sort(). Orders according to the start index.
        If ties, then order according to the end index."""
        if self.start == other.start:
            return self.end < other.end
        return self.start < other.start

    @classmethod
    def cluster_adjacent_spans(cls, spans: List['Span']) -> List[List['Span']]:
        """Cluster adjacent spans like (0,1), (2,3), (4,5).
        This function reorganizes input spans into their own List if they're adjacent.
        """


    @classmethod
    def small_spans_to_big_span(cls, spans: List['Span']) -> 'Span':
        # TODO: add warning for unsorted spans or not-contiguous spans
        # TODO: what happens when Boxes cant be merged?
        start = None
        end = None
        for span in spans:
            if span.start < start:
                start = span.start
            if span.end > end:
                end = span.end
        return Span(
            start=start,
            end=end,
            box=Box.small_boxes_to_big_box(boxes=[span.box for span in spans])
        )
