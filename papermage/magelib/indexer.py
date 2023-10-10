"""


@chrisw, @kylel

"""


from abc import abstractmethod
from typing import List

import numpy as np
from ncls import NCLS

from .box import Box
from .entity import Entity


class Indexer:
    """Stores an index for a particular collection of Entities.
    Indexes in this library focus on *INTERSECT* relations."""

    @abstractmethod
    def find(self, query: Entity) -> List[Entity]:
        """Returns all matching Entities given a suitable query"""
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
            np.array(starts, dtype=np.int32), np.array(ends, dtype=np.int32), np.array(ids, dtype=np.int32)
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
                match_ids = [
                    matched_id for _start, _end, matched_id in self._index.find_overlap(span.start, span.end)
                ]
                if len(match_ids) > 1:
                    matches = [self._entities[match_id].to_json() for match_id in match_ids]
                    raise ValueError(
                        f"Detected overlap! While processing the Span {span} as part of query Entity {entity.to_json()}, we found that it overlaps with existing Entity(s):\n"
                        + "\n".join([f"\t{i}\t{m} " for i, m in zip(match_ids, matches)])
                    )

    def find(self, query: Entity) -> List[Entity]:
        if not isinstance(query, Entity):
            raise TypeError(f"EntityIndexer only works with `query` that is Entity type")

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


class EntityBoxIndexer(Indexer):
    """
    Manages a data structure for locating overlapping BoxGroups.
    Builds a static nested containment list from BoxGroups
    and accepts other BoxGroups as search probes.

    @kylel
    """

    def __init__(self, entities: List[Entity], allow_overlap: bool = True) -> None:
        self._entities = entities

        self._box_id_to_entity_id = {}
        self._boxes = []
        box_id = 0
        for i, e in enumerate(entities):
            for box in e.boxes:
                self._boxes.append(box)
                self._box_id_to_entity_id[box_id] = i
                box_id += 1

        self._np_boxes_x1 = np.array([b.l for b in self._boxes])
        self._np_boxes_y1 = np.array([b.t for b in self._boxes])
        self._np_boxes_x2 = np.array([b.l + b.w for b in self._boxes])
        self._np_boxes_y2 = np.array([b.t + b.h for b in self._boxes])
        self._np_boxes_page = np.array([b.page for b in self._boxes])

        if not allow_overlap:
            self._ensure_disjoint()

    def _find_overlap_boxes(self, query: Box) -> List[int]:
        x1, y1, x2, y2 = query.xy_coordinates
        mask = (
            (self._np_boxes_x1 <= x2)
            & (self._np_boxes_x2 >= x1)
            & (self._np_boxes_y1 <= y2)
            & (self._np_boxes_y2 >= y1)
            & (self._np_boxes_page == query.page)
        )
        return np.where(mask)[0].tolist()

    def _find_overlap_entities(self, query: Box) -> List[int]:
        return [self._box_id_to_entity_id[box_id] for box_id in self._find_overlap_boxes(query)]

    def _ensure_disjoint(self) -> None:
        """
        Constituent box groups must be fully disjoint.
        Ensure the integrity of the built index.
        """
        for entity in self._entities:
            for box in entity.boxes:
                match_ids = self._find_overlap_entities(query=box)
                if len(match_ids) > 1:
                    matches = [self._entities[match_id].to_json() for match_id in match_ids]
                    raise ValueError(
                        f"Detected overlap! While processing the Box {box} as part of query Entity {entity.to_json()}, we found that it overlaps with existing Entity(s):\n"
                        + "\n".join([f"\t{i}\t{m} " for i, m in zip(match_ids, matches)])
                    )

    def find(self, query: Entity) -> List[Entity]:
        if not isinstance(query, Entity):
            raise TypeError(f"EntityBoxIndexer only works with `query` that is Entity type")

        if not query.boxes:
            return []

        match_ids = []
        for box in query.boxes:
            match_ids.extend(self._find_overlap_entities(query=box))

        return [self._entities[match_id] for match_id in sorted(set(match_ids))]
