"""

Needs to be imported in this order.

@kylel

"""


from papermage.magelib.image import Image
from papermage.magelib.span import Span
from papermage.magelib.box import Box
from papermage.magelib.metadata import Metadata
from papermage.magelib.annotation import Annotation
from papermage.magelib.entity import Entity
from papermage.magelib.indexer import EntitySpanIndexer
from papermage.magelib.document import Document
from papermage.magelib.document import (
    MetadataFieldName, 
    EntitiesFieldName, 
    SymbolsFieldName,
    RelationsFieldName, 
    PagesFieldName, 
    TokensFieldName, 
    RowsFieldName,
    ImagesFieldName
)

__all__ = [
    "Document",
    "Annotation" "Entity",
    "Relation",
    "Span",
    "Box",
    "Image",
    "Metadata",
    "EntitySpanIndexer",
    "ImageFieldName",
    "SymbolsFieldName",
    "MetadataFieldName",
    "EntitiesFieldName",
    "RelationsFieldName",
    "PagesFieldName",
    "TokensFieldName",
    "RowsFieldName",
]
