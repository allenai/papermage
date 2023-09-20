# Predictors

Some rules about Predictors:

* Each Predictor is named after the type of Entities it produces. For example, `sentences` come from the `SentencePredictor`. 
  
* Organization looks like:

```
predictors/
|-- base_predictors/
    |-- hf_predictors.py
    |-- lp_predictors.py
    |-- api_predictors.py
    |-- spacy_predictors.py
    |-- sklearn_predictors.py
|-- block_predictors.py
|-- paragraph_predictors.py
|-- sentence_predictors.py
|-- word_predictors.py
|-- token_predictors.py
```
  * Note that `base_predictors/` contains reusable implementations; users never have to import them, but developers may want to reuse these. Users of the library instead import the desired predictor for a given emitted entity type.


* We try to name our predictors `[Framework][Model][Dataset][Entity]`. 

## `SpanQAPredictor` (Using GPT3 as a Predictor)

The `span_qa.predictor.py` file includes an example of using the `decontext` library to use GPT3 as a predictor. The example involves span-based classification: for example, a a user can highlight a span of text in a paper and ask a question about it. (The span is a field, and the question is metadata on the field.) The predictor runs retrieval over specified the specified units and feeds the question, context, and highlighted span to GPT3 to answer the question. See `tests/test_predictors/test_span_qa_predictor.py` for examples of how this predictor is used.