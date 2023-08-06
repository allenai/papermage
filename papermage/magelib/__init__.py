"""

Needs to be imported in this order.

@kylel

"""


from papermage.magelib.annotation import Annotation
from papermage.magelib.box import Box
from papermage.magelib.document import (
    BlocksLayerName,
    Document,
    EntitiesLayerName,
    ImagesLayerName,
    MetadataLayerName,
    PagesLayerName,
    RelationsLayerName,
    RowsLayerName,
    SymbolsLayerName,
    TokensLayerName,
    WordsLayerName,
)
from papermage.magelib.entity import Entity
from papermage.magelib.image import Image
from papermage.magelib.indexer import EntityBoxIndexer, EntitySpanIndexer
from papermage.magelib.metadata import Metadata
from papermage.magelib.span import Span

__all__ = [
    "Document",
    "Annotation" "Entity",
    "Relation",
    "Span",
    "Box",
    "Image",
    "Metadata",
    "EntitySpanIndexer",
    "EntityBoxIndexer",
    "ImageFieldName",
    "SymbolsLayerName",
    "MetadataLayerName",
    "EntitiesLayerName",
    "RelationsLayerName",
    "PagesLayerName",
    "TokensLayerName",
    "RowsLayerName",
    "BlocksLayerName",
    "WordsLayerName",
]
