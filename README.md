# papermage

### Setup

```python
conda create -n papermage python=3.11
conda activate papermage
pip install -e '.[dev,predictors]'
```

If you're on MacOSX, you'll also want to run:
```
conda install poppler
```


## Unit testing
```bash
python -m pytest
```
for latest failed test
```bash
python -m pytest --lf --no-cov -n0
```
for specific test name of class name
```bash
python -m pytest -k 'TestPDFPlumberParser' --no-cov -n0
```

## Quick start

#### 1. Create a Document for the first time from a PDF
TODO

#### 2. Understanding the output: the `Document` class

What is a `Document`? At minimum, it is some text, saved under the `.symbols` field, which is just a `<str>`.  For example:

```python
doc.symbols
> "Language Models as Knowledge Bases?\nFabio Petroni1 Tim Rockt..."
```

But this library is really useful when you have multiple different ways of segmenting `.symbols`. For example, segmenting the paper into Pages, and then each page into Rows:

```python
for page in doc.pages:
    print(f'\n=== PAGE: {page.id} ===\n\n')
    for row in page.rows:
        print(row.symbols)
        
> ...
> === PAGE: 5 ===
> ['tence x, s′ will be linked to s and o′ to o. In']
> ['practice, this means RE can return the correct so-']
> ['lution o if any relation instance of the right type']
> ['was extracted from x, regardless of whether it has']
> ...
```

This shows two nice aspects of this library:

* `Document` provides iterables for different segmentations of `symbols`.  Options include things like `pages, tokens, rows, sents, paragraphs, sections, ...`.  Not every Parser will provide every segmentation, though.

* Each one of these segments (in our library, we call them `Entity` objects) is aware of (and can access) other segment types. For example, you can call `page.rows` to get all Rows that intersect a particular Page. Or you can call `sent.tokens` to get all Tokens that intersect a particular Sentence. Or you can call `sent.rows` to get the Row(s) that intersect a particular Sentence. These indexes are built *dynamically* when the `Document` is created and each time a new `Entity` type is added. In the extreme, as long as those fields are available in the Document, you can write:

```python
for page in doc.pages:
    for paragraph in page.paragraphs:
        for sent in paragraph.sents:
            for row in sent.rows: 
                ...
```

You can check which fields are available in a Document via:

```python
doc.fields
> ['pages', 'tokens', 'rows']
```

#### 3. Understanding intersection of Entities

Note that `Entity`s don't necessarily perfectly nest each other. For example, what happens if you run:

```python
for sent in doc.sents:
    for row in sent.rows:
        print([token.symbols for token in row.tokens])
```

Tokens that are *outside* each sentence can still be printed. This is because when we jump from a sentence to its rows, we are looking for *all* rows that have *any* overlap with the sentence. Rows can extend beyond sentence boundaries, and as such, can contain tokens outside that sentence.

Here's another example:
```python
for page in doc.pages:
    print([sent.symbols for sent in page.sents])
```

Sentences can cross page boundaries. As such, adjacent pages may end up printing the same sentence.

But rows and tokens adhere strictly to page boundaries, and thus will not repeat when printed across pages:
```python
for page in doc.pages:
    print([row.symbols for row in page.rows])
    print([token.symbols for token in page.tokens])
``` 

A key aspect of using this library is understanding how these different fields are defined & anticipating how they might interact with each other. We try to make decisions that are intuitive, but we do ask users to experiment with fields to build up familiarity.



#### 4. What's in an `Entity`?

Each `Entity` object stores information about its contents and position:

* `.spans: List[Span]`, A `Span` is a pointer into `Document.symbols` (that is, `Span(start=0, end=5)` corresponds to `symbols[0:5]`). By default, when you iterate over an `Entity`, you iterate over its `.spans`.

* `.boxes: List[Box]`, A `Box` represents a rectangular region on the page. Each span is associated a Box.

* `.metadata: Metadata`, A free form dictionary-like object to store extra metadata about that `Entity`. These are usually empty.



#### 5. How can I manually create my own `Document`?

A `Document` is created by stitching together 3 types of tools: `Parsers`, `Rasterizers` and `Predictors`.

* `Parsers` take a PDF as input and return a `Document` compared of `.symbols` and other fields. The example one we use is a wrapper around [PDFPlumber](https://github.com/jsvine/pdfplumber) - MIT License utility.

* `Rasterizers` take a PDF as input and return an `Image` per page that is added to `Document.images`. The example one we use is [PDF2Image](https://github.com/Belval/pdf2image) - MIT License. 

* `Predictors` take a `Document` and apply some operation to compute a new set of `Entity` objects that we can insert into our `Document`. These are all built in-house and can be either simple heuristics or full machine-learning models.



#### 6. How can I save my `Document`?

```python
import json
with open('filename.json', 'w') as f_out:
    json.dump(doc.to_json(with_images=True), f_out, indent=4)
```

will produce something akin to:
```python
{
    "symbols": "Language Models as Knowledge Bases?\nFabio Petroni1 Tim Rockt...",
    "entities": {
        "images": [...],
        "rows": [...],
        "tokens": [...],
        "words": [...],
        "blocks": [...],
        "vila_span_groups": [...]
    },
    "relations": {...},
    "metadata": {...}
}
```

Note that `Images` are serialized to `base64` if you include `with_images` flag. Otherwise, it's left out of JSON serialization by default.


#### 7. How can I load my `Document`?

These can be used to reconstruct a `Document` again via:

```python
with open('filename.json') as f_in:
    doc_dict = json.load(f_in)
    doc = Document.from_json(doc_dict)
```


Note: A common pattern for adding fields to a document is to load in a previously saved document, run some additional `Predictors` on it, and save the result.

See `papermage/predictors/README.md` for more information about training customer predictors on your own data.