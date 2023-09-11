"""

A (start, end) interval that references some string. For example, span (0, 4)
represents the word 'This' in the string 'This is a document.'

@kylel, @egork

"""

from collections import defaultdict
from typing import Dict, List


class Span:
    __slots__ = ["start", "end"]

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
        return f"Span{self.to_json()}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Span):
            return False
        return self.start == other.start and self.end == other.end

    def __lt__(self, other: "Span"):
        """Useful for sort(). Orders according to the start index.
        If ties, then order according to the end index."""
        if self.start == other.start:
            return self.end < other.end
        return self.start < other.start

    def __hash__(self) -> int:
        return hash((self.start, self.end))

    def is_overlap(self, other: "Span") -> bool:
        """Whether self overlaps with the other Span object."""
        return self.start <= other.start < self.end or other.start <= self.start < other.end or self == other

    @classmethod
    def create_enclosing_span(cls, spans: List["Span"]) -> "Span":
        """Create the narrowest Span that completely encloses all the input Spans."""
        if not spans:
            raise ValueError(f"`spans` should be non-empty.")
        start = spans[0].start
        end = spans[0].end
        for span in spans[1:]:
            if span.start < start:
                start = span.start
            if span.end > end:
                end = span.end
        return Span(start=start, end=end)
