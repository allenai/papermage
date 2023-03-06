"""


@kylel

"""

import itertools
import warnings
from typing import Dict, Iterable, List, Optional

from papermage.types import Entity, Span, Box, Image, Metadata, EntitySpanIndexer

SymbolsFieldName = 'symbols'
ImagesFieldName = 'images'
MetadataFieldName = 'metadata'
EntitiesFieldName = 'entities'
RelationsFieldName = 'relations'


class Document:
    SPECIAL_FIELDS = [SymbolsFieldName,
                      ImagesFieldName,
                      MetadataFieldName,
                      EntitiesFieldName,
                      RelationsFieldName]

    def __init__(self,
                 symbols: str,
                 metadata: Optional[Metadata] = None):
        self.symbols = symbols
        self.metadata = metadata if metadata else Metadata()
        self.__entity_span_indexers: Dict[str, EntitySpanIndexer] = {}

    def find_span_overlap_entities(self, query: Entity, field_name: str) -> List[Entity]:
        return self.__entity_span_indexers[field_name].find(query=query)

    def check_field_name_availability(self, field_name: str) -> None:
        if field_name in self.SPECIAL_FIELDS:
            raise AssertionError(f"{field_name} not allowed Document.SPECIAL_FIELDS.")
        if field_name in self.__entity_span_indexers.keys():
            raise AssertionError(f"{field_name} already exists. Try `is_overwrite=True`")
        if field_name in dir(self):
            raise AssertionError(f"{field_name} clashes with Document class properties.")

    def annotate_entity(self, field_name: str, entities: List[Entity]) -> None:
        self.check_field_name_availability(field_name=field_name)

        for entity in entities:
            entity.doc = self

        self.__entity_span_indexers[field_name] = EntitySpanIndexer(entities=entities)

        setattr(self, field_name, entities)
        self.__entities.append(field_name)

    def remove_entity(self, field_name: str):
        delattr(self, field_name)
        self.__entities = [f for f in self.__entities if f != field_name]
        del self.__entity_span_indexers[field_name]

    def to_json(self, field_names: Optional[List[str]] = None) -> Dict:
        """Returns a dictionary that's suitable for serialization

        Use `fields` to specify a subset of groups in the Document to include (e.g. 'sentences')

        Output format looks like
            {
                symbols: "...",
                entities: {...},
                relations: {...},
                metadata: {...}
            }
        """
        # 1) instantiate basic Document dict
        doc_dict = {
            SymbolsFieldName: self.symbols,
            MetadataFieldName: self.metadata.to_json(),
            EntitiesFieldName: {},
            RelationsFieldName: {}
        }

        # 2) serialize each field to JSON
        field_names = self.__entities if field_names is None else field_names
        for field_name in field_names:
            doc_dict[EntitiesFieldName][field_name] = [
                entity.to_json() for entity in getattr(self, field_name)
            ]

        return doc_dict

    @classmethod
    def from_json(cls, doc_json: Dict) -> "Document":
        # 1) instantiate basic Document
        symbols = doc_json[SymbolsFieldName]
        doc = cls(symbols=symbols, metadata=Metadata(**doc_json.get(MetadataFieldName, {})))

        # 2) instantiate entities
        for field_name, entity_jsons in doc_json[EntitiesFieldName].items():
            entities = [
                Entity.from_json(entity_json=entity_json)
                for entity_json in entity_jsons
            ]
            doc.annotate_entity(field_name=field_name, entities=entities)

        return doc
