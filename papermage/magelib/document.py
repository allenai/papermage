"""


@kylel

"""

from itertools import chain
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

from .box import Box
from .entity import Entity
from .image import Image
from .indexer import EntityBoxIndexer, EntitySpanIndexer
from .layer import Layer
from .metadata import Metadata
from .span import Span

# document field names
SymbolsFieldName = "symbols"
ImagesFieldName = "images"
MetadataFieldName = "metadata"
EntitiesFieldName = "entities"
RelationsFieldName = "relations"

PagesFieldName = "pages"
TokensFieldName = "tokens"
RowsFieldName = "rows"
BlocksFieldName = "blocks"
WordsFieldName = "words"
SentencesFieldName = "sentences"
ParagraphsFieldName = "paragraphs"

# these come from vila
TitlesFieldName = "titles"
AuthorsFieldName = "authors"
AbstractsFieldName = "abstracts"
KeywordsFieldName = "keywords"
SectionsFieldName = "sections"
ListsFieldName = "lists"
BibliographiesFieldName = "bibliographies"
EquationsFieldName = "equations"
AlgorithmsFieldName = "algorithms"
FiguresFieldName = "figures"
TablesFieldName = "tables"
CaptionsFieldName = "captions"
HeadersFieldName = "headers"
FootersFieldName = "footers"
FootnotesFieldName = "footnotes"


class Prediction(NamedTuple):
    name: str
    entities: List[Entity]


class Document:
    tokens: Layer
    rows: Layer
    blocks: Layer
    words: Layer
    sentences: Layer
    paragraphs: Layer
    pages: Layer

    SPECIAL_FIELDS = [SymbolsFieldName, ImagesFieldName, MetadataFieldName]

    def __init__(
        self,
        symbols: Optional[str] = None,
        images: Optional[List[Image]] = None,
        metadata: Optional[Metadata] = None,
    ):
        self.symbols = symbols if symbols else None
        self.images = images if images else None
        if not self.symbols and not self.images:
            raise ValueError("Document must have at least one of `symbols` or `images`")
        self.metadata = metadata if metadata else Metadata()
        self.__entity_span_indexers: Dict[str, EntitySpanIndexer] = {}
        self.__entity_box_indexers: Dict[str, EntityBoxIndexer] = {}

    @property
    def layers(self) -> List[str]:
        return self.SPECIAL_FIELDS + list(self.__entity_span_indexers.keys())

    def find(self, query: Union[Span, Box], field_name: str) -> List[Entity]:
        if isinstance(query, Span):
            return self.intersect_by_span(query=Entity(spans=[query]), field_name=field_name)
            # return self.__entity_span_indexers[field_name].find(query=Entity(spans=[query]))
        elif isinstance(query, Box):
            return self.intersect_by_box(query=Entity(boxes=[query]), field_name=field_name)
            # return self.__entity_box_indexers[field_name].find(query=Entity(boxes=[query]))
        else:
            raise TypeError(f"Unsupported query type {type(query)}")

    def intersect_by_span(self, query: Entity, field_name: str) -> List[Entity]:
        return self.__entity_span_indexers[field_name].find(query=query)

    def intersect_by_box(self, query: Entity, field_name: str) -> List[Entity]:
        return self.__entity_box_indexers[field_name].find(query=query)

    def validate_layer_name_availability(self, name: str) -> None:
        if name in self.SPECIAL_FIELDS:
            raise AssertionError(f"{name} not allowed Document.SPECIAL_FIELDS.")
        if name in self.__entity_span_indexers.keys():
            raise AssertionError(f'{name} already exists. Try `doc.remove_entity("{name}")` first.')
        if name in dir(self):
            raise AssertionError(f"{name} clashes with Document class properties.")

    def get_layer(self, name: str) -> Layer:
        """Gets a layer by name. For example, `doc.get_layer("sentences")` returns sentences."""
        return getattr(self, name)

    def annotate(self, *predictions: Union[Prediction, Tuple[Prediction, ...]]) -> None:
        """Annotates the document with predictions."""
        all_preds = chain.from_iterable([p] if isinstance(p, Prediction) else p for p in predictions)
        for prediction in all_preds:
            self.annotate_layer(name=prediction.name, entities=prediction.entities)

    def annotate_layer(self, name: str, entities: List[Entity]) -> None:
        self.validate_layer_name_availability(name=name)

        for i, entity in enumerate(entities):
            entity.doc = self
            entity.id = i

        self.__entity_span_indexers[name] = EntitySpanIndexer(entities=entities)
        self.__entity_box_indexers[name] = EntityBoxIndexer(entities=entities)
        setattr(self, name, entities)

    def remove_layer(self, name: str):
        for entity in self.get_layer(name=name):
            entity.doc = None

        delattr(self, name)
        del self.__entity_span_indexers[name]
        del self.__entity_box_indexers[name]

    def get_relation(self, name: str) -> List["Relation"]:
        raise NotImplementedError

    def annotate_relation(self, name: str) -> None:
        self.validate_layer_name_availability(name=name)
        raise NotImplementedError

    def remove_relation(self, name: str) -> None:
        raise NotImplementedError

    def annotate_images(self, images: List[Image]) -> None:
        if len(images) == 0:
            raise ValueError("No images were provided")

        image_types = {type(image) for image in images}
        if len(image_types) > 1:
            raise TypeError(f"Images contain multiple types: {image_types}")
        image_type = image_types.pop()

        if not issubclass(image_type, Image):
            raise NotImplementedError(f"Unsupported image type {image_type} for {ImagesFieldName}")

        setattr(self, ImagesFieldName, images)

    def remove_images(self) -> None:
        raise NotImplementedError

    def to_json(self, field_names: Optional[List[str]] = None, with_images: bool = False) -> Dict:
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
            RelationsFieldName: {},
        }

        # 2) serialize each field to JSON
        field_names = list(self.__entity_span_indexers.keys()) if field_names is None else field_names
        for field_name in field_names:
            doc_dict[EntitiesFieldName][field_name] = [entity.to_json() for entity in getattr(self, field_name)]

        # 3) serialize images if `with_images == True`
        if with_images:
            doc_dict[ImagesFieldName] = [image.to_base64() for image in getattr(self, ImagesFieldName)]

        return doc_dict

    @classmethod
    def from_json(cls, doc_json: Dict) -> "Document":
        # 1) instantiate basic Document
        symbols = doc_json[SymbolsFieldName]
        doc = cls(symbols=symbols, metadata=Metadata(**doc_json.get(MetadataFieldName, {})))

        # 2) instantiate entities
        for field_name, entity_jsons in doc_json[EntitiesFieldName].items():
            entities = [Entity.from_json(entity_json=entity_json) for entity_json in entity_jsons]
            doc.annotate_layer(name=field_name, entities=entities)

        return doc
