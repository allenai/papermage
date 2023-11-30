"""

Paragraph Predictor

@kylel

"""

import re
from typing import List, Tuple

from papermage.magelib import (
    BlocksFieldName,
    Document,
    Entity,
    PagesFieldName,
    ParagraphsFieldName,
    RowsFieldName,
    Span,
    TokensFieldName,
)
from papermage.predictors import BasePredictor
