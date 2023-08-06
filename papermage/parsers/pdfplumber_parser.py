import itertools
import string
from typing import List, Optional, Tuple, Union

import pdfplumber

try:
    # pdfplumber >= 0.8.0
    import pdfplumber.utils.text as ppu
except:
    # pdfplumber <= 0.7.6
    import pdfplumber.utils as ppu

from papermage.magelib import (
    Box,
    Document,
    EntitiesFieldName,
    Entity,
    Metadata,
    PagesFieldName,
    RowsFieldName,
    Span,
    SymbolsFieldName,
    TokensFieldName,
)
from papermage.parsers.parser import Parser
from papermage.utils.text import maybe_normalize

_TOL = Union[int, float]


class WordExtractorWithFontInfo(ppu.WordExtractor):
    """Override WordExtractor methods to append additional char-level info."""

    def __init__(
        self,
        x_tolerance: _TOL = ppu.DEFAULT_X_TOLERANCE,
        y_tolerance: _TOL = ppu.DEFAULT_Y_TOLERANCE,
        keep_blank_chars: bool = False,
        use_text_flow: bool = False,
        horizontal_ltr: bool = True,
        vertical_ttb: bool = True,
        extra_attrs: Optional[List[str]] = None,
        split_at_punctuation: Union[bool, str] = False,
        append_attrs: Optional[List[str]] = None,
    ):
        super().__init__(
            x_tolerance=x_tolerance,
            y_tolerance=y_tolerance,
            keep_blank_chars=keep_blank_chars,
            use_text_flow=use_text_flow,
            horizontal_ltr=horizontal_ltr,
            vertical_ttb=vertical_ttb,
            extra_attrs=extra_attrs,
            split_at_punctuation=split_at_punctuation,
        )
        self.append_attrs = append_attrs

    def merge_chars(self, ordered_chars: ppu.T_obj_list) -> ppu.T_obj:
        """Modify returned word to capture additional char-level info."""
        word = super().merge_chars(ordered_chars=ordered_chars)

        # ordered_chars is a list of characters for the word with extra attributes.
        # Here we simply append additional information to the word using the first char.
        if self.append_attrs is not None:
            for key in self.append_attrs:
                if key not in word:
                    word[key] = ordered_chars[0][key]

        return word

    def extract(self, chars: ppu.T_obj_list) -> ppu.T_obj_list:
        if hasattr(super(), "extract"):
            # pdfplumber <= 0.7.6
            return super().extract(chars)
        else:
            # pdfplumber >= 0.8.0
            return super().extract_words(chars)


class PDFPlumberParser(Parser):
    # manually added characters: '–' and '§'
    DEFAULT_PUNCTUATION_CHARS = string.punctuation + chr(8211) + chr(167)

    def __init__(
        self,
        token_x_tolerance: _TOL = 1.5,
        token_y_tolerance: _TOL = 2,
        line_x_tolerance: _TOL = 10,
        line_y_tolerance: _TOL = 10,
        keep_blank_chars: bool = False,
        use_text_flow: bool = True,
        horizontal_ltr: bool = True,
        vertical_ttb: bool = True,
        extra_attrs: Optional[List[str]] = None,
        split_at_punctuation: Union[str, bool] = True,
    ):
        """The PDFPlumber PDF Detector

        Args:
            token_x_tolerance (_TOL, optional):
                The threshold used for extracting "word tokens" from the pdf file.
                It will merge the pdf characters into a word token if the difference
                between the x_2 of one character and the x_1 of the next is less than
                or equal to token_x_tolerance. See details in `pdf2plumber's documentation
                <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
                Defaults to 1.5, in absolute coordinates.
            token_y_tolerance (_TOL, optional):
                The threshold used for extracting "word tokens" from the pdf file.
                It will merge the pdf characters into a word token if the difference
                between the y_2 of one character and the y_1 of the next is less than
                or equal to token_y_tolerance. See details in `pdf2plumber's documentation
                <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
                Defaults to 2, in absolute coordinates.
            line_x_tolerance (_TOL, optional):
                The threshold used for extracting "line tokens" from the pdf file.
                Defaults to 10, in absolute coordinates.
            line_y_tolerance (_TOL, optional):
                The threshold used for extracting "line tokens" from the pdf file.
                Defaults to 10, in absolute coordinates.
            keep_blank_chars (bool, optional):
                When keep_blank_chars is set to True, it will treat blank characters
                are treated as part of a word, not as a space between words. See
                details in `pdf2plumber's documentation
                <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
                Defaults to False.
            use_text_flow (bool, optional):
                When use_text_flow is set to True, it will use the PDF's underlying
                flow of characters as a guide for ordering and segmenting the words,
                rather than presorting the characters by x/y position. (This mimics
                how dragging a cursor highlights text in a PDF; as with that, the
                order does not always appear to be logical.) See details in
                `pdf2plumber's documentation
                <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
                Defaults to True.
            horizontal_ltr (bool, optional):
                When horizontal_ltr is set to True, it means the doc should read
                text from left to right, vice versa.
                Defaults to True.
            vertical_ttb (bool, optional):
                When vertical_ttb is set to True, it means the doc should read
                text from top to bottom, vice versa.
                Defaults to True.
            extra_attrs (Optional[List[str]], optional):
                Passing a list of extra_attrs (e.g., ["fontname", "size"]) will
                restrict each words to characters that share exactly the same
                value for each of those `attributes extracted by pdfplumber
                <https://github.com/jsvine/pdfplumber/blob/develop/README.md#char-properties>`_,
                and the resulting word dicts will indicate those attributes.
                See details in `pdf2plumber's documentation
                <https://github.com/jsvine/pdfplumber#the-pdfplumberpage-class>`_.
                Defaults to `["fontname", "size"]`.
        """
        self.token_x_tolerance = token_x_tolerance
        self.token_y_tolerance = token_y_tolerance
        self.line_x_tolerance = line_x_tolerance
        self.line_y_tolerance = line_y_tolerance
        self.keep_blank_chars = keep_blank_chars
        self.use_text_flow = use_text_flow
        self.horizontal_ltr = horizontal_ltr
        self.vertical_ttb = vertical_ttb
        self.extra_attrs = extra_attrs if extra_attrs is not None else ["fontname", "size"]
        if split_at_punctuation is True:
            split_at_punctuation = type(self).DEFAULT_PUNCTUATION_CHARS
        self.split_at_punctuation = split_at_punctuation

    def parse(self, input_pdf_path: str, **kwargs) -> Document:
        with pdfplumber.open(input_pdf_path) as plumber_pdf_object:
            all_tokens = []
            all_word_ids = []
            last_word_id = -1
            all_row_ids = []
            last_row_id = -1
            all_page_ids = []
            all_page_dims = []

            for page_id, page in enumerate(plumber_pdf_object.pages):
                page_unit = float(page.page_obj.attrs.get("UserUnit", 1.0))
                all_page_dims.append((float(page.width), float(page.height), page_unit))

                # 1) tokens we use for Document.symbols
                coarse_tokens = page.extract_words(
                    x_tolerance=self.token_x_tolerance,
                    y_tolerance=self.token_y_tolerance,
                    keep_blank_chars=self.keep_blank_chars,
                    use_text_flow=self.use_text_flow,
                    horizontal_ltr=self.horizontal_ltr,
                    vertical_ttb=self.vertical_ttb,
                    extra_attrs=self.extra_attrs,
                    split_at_punctuation=None,
                )
                # 2) tokens we use for Document.tokens
                fine_tokens = WordExtractorWithFontInfo(
                    x_tolerance=self.token_x_tolerance,
                    y_tolerance=self.token_y_tolerance,
                    keep_blank_chars=self.keep_blank_chars,
                    use_text_flow=self.use_text_flow,
                    horizontal_ltr=self.horizontal_ltr,
                    vertical_ttb=self.vertical_ttb,
                    extra_attrs=self.extra_attrs,
                    split_at_punctuation=self.split_at_punctuation,
                    append_attrs=["fontname", "size"],
                ).extract(page.chars)
                # 3) align fine tokens with coarse tokens
                word_ids_of_fine_tokens = self._align_coarse_and_fine_tokens(
                    coarse_tokens=[c["text"] for c in coarse_tokens],
                    fine_tokens=[f["text"] for f in fine_tokens],
                )
                assert len(word_ids_of_fine_tokens) == len(fine_tokens)
                # 4) normalize / clean tokens & boxes
                fine_tokens = [
                    {
                        "text": token["text"],
                        "fontname": maybe_normalize(token["fontname"]),
                        "size": token["size"],
                        "bbox": Box.from_xy_coordinates(
                            x1=float(token["x0"]),
                            y1=float(token["top"]),
                            x2=float(token["x1"]),
                            y2=float(token["bottom"]),
                            page_width=float(page.width),
                            page_height=float(page.height),
                            page=int(page_id),
                        ).to_relative(page_width=float(page.width), page_height=float(page.height)),
                    }
                    for token in fine_tokens
                ]
                # 5) group tokens into lines
                # TODO - doesnt belong in parser; should be own predictor
                line_ids_of_fine_tokens = self._simple_line_detection(
                    page_tokens=fine_tokens,
                    x_tolerance=self.line_x_tolerance / page.width,
                    y_tolerance=self.line_y_tolerance / page.height,
                )
                assert len(line_ids_of_fine_tokens) == len(fine_tokens)
                # 6) accumulate
                all_tokens.extend(fine_tokens)
                all_row_ids.extend([i + last_row_id + 1 for i in line_ids_of_fine_tokens])
                last_row_id = all_row_ids[-1]
                all_word_ids.extend([i + last_word_id + 1 for i in word_ids_of_fine_tokens])
                last_word_id = all_word_ids[-1]
                for _ in fine_tokens:
                    all_page_ids.append(page_id)
            # now turn into a beautiful document!
            doc_json = self._convert_nested_text_to_doc_json(
                token_dicts=all_tokens,
                word_ids=all_word_ids,
                row_ids=all_row_ids,
                page_ids=all_page_ids,
                dims=all_page_dims,
            )
            doc = Document.from_json(doc_json)
            return doc

    def _convert_nested_text_to_doc_json(
        self,
        token_dicts: List[dict],
        word_ids: List[int],
        row_ids: List[int],
        page_ids: List[int],
        dims: List[Tuple[float, float, float]],
    ) -> dict:
        """For a single page worth of text"""

        # 1) build tokens & symbols
        symbols = ""
        token_annos: List[Entity] = []
        start = 0
        for token_id in range(len(token_dicts) - 1):
            token_dict = token_dicts[token_id]
            current_word_id = word_ids[token_id]
            next_word_id = word_ids[token_id + 1]
            current_row_id = row_ids[token_id]
            next_row_id = row_ids[token_id + 1]

            # 1) add to symbols
            symbols += token_dict["text"]

            # 2) make Token
            end = start + len(token_dict["text"])

            # 2b) gather token metadata
            if "fontname" in token_dict and "size" in token_dict:
                token_metadata = Metadata(
                    fontname=token_dict["fontname"],
                    size=token_dict["size"],
                )
            else:
                token_metadata = None

            token = Entity(
                spans=[Span(start=start, end=end)],
                boxes=[token_dict["bbox"]],
                metadata=token_metadata,
            )
            token_annos.append(token)

            # 3) increment whitespace based on Row & Word membership. and build Rows.
            if next_row_id == current_row_id:
                if next_word_id == current_word_id:
                    start = end
                else:
                    symbols += " "
                    start = end + 1
            else:
                # new row
                symbols += "\n"
                start = end + 1
        # handle last token
        symbols += token_dicts[-1]["text"]
        end = start + len(token_dicts[-1]["text"])
        token = Entity(
            spans=[Span(start=start, end=end)],
            boxes=[token_dicts[-1]["bbox"]],
        )
        token_annos.append(token)

        # 2) build rows
        tokens_with_group_ids = [
            (token, row_id, page_id) for token, row_id, page_id in zip(token_annos, row_ids, page_ids)
        ]
        row_annos: List[Entity] = []
        for row_id, tups in itertools.groupby(iterable=tokens_with_group_ids, key=lambda tup: tup[1]):
            row_tokens = [token for token, _, _ in tups]
            row = Entity(
                spans=[
                    Span(
                        start=row_tokens[0].spans[0].start,
                        end=row_tokens[-1].spans[0].end,
                    )
                ],
                boxes=[Box.create_enclosing_box(boxes=[box for t in row_tokens for box in t.boxes])],
            )
            row_annos.append(row)

        # 3) build pages
        page_annos: List[Entity] = []
        for page_id, tups in itertools.groupby(iterable=tokens_with_group_ids, key=lambda tup: tup[2]):
            page_tokens = [token for token, _, _ in tups]
            page_w, page_h, page_unit = dims[page_id]
            page = Entity(
                spans=[
                    Span(
                        start=page_tokens[0].spans[0].start,
                        end=page_tokens[-1].spans[0].end,
                    )
                ],
                boxes=[Box.create_enclosing_box(boxes=[box for t in page_tokens for box in t.boxes])],
                metadata=Metadata(width=page_w, height=page_h, user_unit=page_unit),
            )
            page_annos.append(page)

        return {
            SymbolsFieldName: symbols,
            EntitiesFieldName: {
                TokensFieldName: [token.to_json() for token in token_annos],
                RowsFieldName: [row.to_json() for row in row_annos],
                PagesFieldName: [page.to_json() for page in page_annos],
            },
        }

    def _simple_line_detection(
        self, page_tokens: List[dict], x_tolerance: int = 10, y_tolerance: int = 10
    ) -> List[int]:
        """Get text lines from the page_tokens.
        It will automatically add new lines for 1) line breaks (i.e., the current token
        has a larger y_difference between the previous one than the y_tolerance) or
        2) big horizontal gaps (i.e., the current token has a larger y_difference between
        the previous one than the x_tolerance)

        Adapted from https://github.com/allenai/VILA/blob/e6d16afbd1832f44a430074855fbb4c3d3604f4a/src/vila/pdftools/pdfplumber_extractor.py#L24

        Modified Oct 2022 (kylel): Changed return value to be List[int]
        """
        prev_y = None
        prev_x = None

        lines = []
        cur_line_id = 0
        n = 0

        for token in page_tokens:
            cur_y = token["bbox"].center[1]
            cur_x = token["bbox"].xy_coordinates[0]

            if prev_y is None:
                prev_y = cur_y
                prev_x = cur_x

            if abs(cur_y - prev_y) <= y_tolerance and cur_x - prev_x <= x_tolerance:
                lines.append(cur_line_id)
                if n == 0:
                    prev_y = cur_y
                else:
                    prev_y = (prev_y * n + cur_y) / (n + 1)  # EMA of the y_height
                n += 1

            else:
                cur_line_id += 1

                lines.append(cur_line_id)
                n = 1
                prev_y = cur_y

            prev_x = token["bbox"].xy_coordinates[2]

        return lines

    def _align_coarse_and_fine_tokens(self, coarse_tokens: List[str], fine_tokens: List[str]) -> List[int]:
        """Returns a list of length len(fine_tokens) where elements of the list are
        integer indices into coarse_tokens elements."""
        assert len(coarse_tokens) <= len(fine_tokens), f"This method requires |coarse| <= |fine|"
        assert "".join(coarse_tokens) == "".join(
            fine_tokens
        ), f"This method requires the chars(coarse) == chars(fine)"

        coarse_start_ends = []
        start = 0
        for token in coarse_tokens:
            end = start + len(token)
            coarse_start_ends.append((start, end))
            start = end

        fine_start_ends = []
        start = 0
        for token in fine_tokens:
            end = start + len(token)
            fine_start_ends.append((start, end))
            start = end

        fine_id = 0
        coarse_id = 0
        out = []
        while fine_id < len(fine_start_ends) and coarse_id < len(coarse_start_ends):
            fine_start, fine_end = fine_start_ends[fine_id]
            coarse_start, coarse_end = coarse_start_ends[coarse_id]
            if coarse_start <= fine_start and fine_end <= coarse_end:
                out.append(coarse_id)
                fine_id += 1
            else:
                coarse_id += 1

        return out
