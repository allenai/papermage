"""


@chrisw, @kylel

"""


from typing import List

from abc import abstractmethod

import numpy as np
from papermage.types import Entity, Annotation
from ncls import NCLS


class Indexer:
    """Stores an index for a particular collection of Annotations.
    Indexes in this library focus on *INTERSECT* relations."""

    @abstractmethod
    def find(self, query: Annotation) -> List[Annotation]:
        """Returns all matching Annotations given a suitable query"""
        raise NotImplementedError()


class EntitySpanIndexer(Indexer):
    """
    Manages a data structure for locating overlapping Entity Spans.
    Builds a static nested containment list from Entity Spans
    and accepts other Entity as search probes.

    See: https://github.com/biocore-ntnu/ncls

    [citation]
    Alexander V. Alekseyenko, Christopher J. Lee;
    Nested Containment List (NCList): a new algorithm for accelerating interval query of genome
      alignment and interval databases, Bioinformatics,
    Volume 23, Issue 11, 1 June 2007, Pages 1386–1393, https://doi.org/10.1093/bioinformatics/btl647
    """

    def __init__(self, entities: List[Entity]) -> None:
        starts = []
        ends = []
        ids = []

        for id, entity in enumerate(entities):
            for span in entity.spans:
                starts.append(span.start)
                ends.append(span.end)
                ids.append(id)

        self._entities = entities
        self._index = NCLS(
            np.array(starts, dtype=np.int32),
            np.array(ends, dtype=np.int32),
            np.array(ids, dtype=np.int32)
        )

        self._ensure_disjoint()

    def _ensure_disjoint(self) -> None:
        """
        Constituent span groups must be fully disjoint.
        Ensure the integrity of the built index.
        """
        for entity in self._entities:
            for span in entity.spans:
                matches = [match for match in self._index.find_overlap(span.start, span.end)]
                if len(matches) > 1:
                    raise ValueError(
                        f"Detected overlap with existing Entity(s) {matches} for {entity}"
                    )

    def find(self, query: Entity) -> List[Entity]:
        if not isinstance(query, Entity):
            raise ValueError(f'EntityIndexer only works with `query` that is Entity type')

        if not query.spans:
            return []

        matched_ids = set()

        for span in query.spans:
            for _start, _end, matched_id in self._index.find_overlap(span.start, span.end):
                matched_ids.add(matched_id)

        matched_entities = [self._entities[matched_id] for matched_id in matched_ids]

        # TODO: kylel - is this necessary? seems like already does this (see tests)
        # Retrieval above doesn't preserve document order; sort here
        # TODO: provide option to return matched span groups in same order as self._entities
        #   (the span groups the index was built with originally)
        return sorted(list(matched_entities))

