# Trainers

## Training custom predictors (`entity_classification_predictor`)
To train a custom predictor, first collect data and format it into a `Document` where one of the fields is the collected data.
For example, let's say, we wanted to make a predictor that predicted if a word started with the letter "t" or "d". (Of course, it would be silly to train a classifier to do this, but this is just an example.)

### Formatting Data
First collect some data, and put it in `Document` objects:

```
doc_1 = Document.from_json({
    "symbols": "This is a test docudock that is not very long.",
    "entities": {
        "tokens": [...],
        "rows": [...],
        "pages": [...],
        "words_starting_with_td": [word.to_json() for word in [
            Entity(spans=[Span(start=0, end=4)]),
            Entity(spans=[Span(start=10, end=14)]),
            Entity(spans=[Span(start=15, end=23)]),
            Entity(spans=[Span(start=24, end=28)]),
        ]]
    }
})

doc_2 = Document.from_json({
    "symbols": "This is a second test docudock that is also quite short.",
    "entities": {
        "tokens": [...],
        "rows": [...],
        "pages": [...],
        "words_starting_with_td": [word.to_json() for word in [
            Entity(spans=[Span(start=0, end=4)]),
            Entity(spans=[Span(start=17, end=21)]),
            Entity(spans=[Span(start=22, end=30)]),
            Entity(spans=[Span(start=31, end=35)]),
        ]]
    }
})
```

These should then be saved to a `jsonl` file with one document per line. (For example, see `tests/fixtures/predictor_training_docs_tiny.jsonl`).

### Run the Trainer
The trainer can be run at the command line with:
```bash
python papermage/trainers/entity_classification_predictor_trainer.py data_path={path/to/data} label_field={field with the label} param1=value1 ...
```
where
* `data_path` is the path to the `jsonl` file containing the annotated documents
* `label_field` is the name of the field (e.g. `words_starting_with_td`)
* `param`s are additional training parameters. They are explained in the `EntityClassificationTrainConfig` class in `papermage/trainers/entity_classification_predictor_trainer.py`. We use the pytorch-lightning [`Trainer`](https://lightning.ai/docs/pytorch/latest/api/lightning.pytorch.trainer.trainer.Trainer.html#lightning.pytorch.trainer.trainer.Trainer), so these parameters can be any keyword arguments to the class.

### Loading the Predictor
By default, the model checkpoints are saved to a subdirectory of `$HOME/.cache/papermage`, but this can be changed by setting the envieronment variable `export PAPERMAGE_CACHE_DIR=path/to/cache` environment variable or by specifying the `default_root_dir` when training.
The name of the subdirectory is computed based on the training arguments and is logged during training.

This path can be passed to `EntityClassificationPredictor.from_pretrained` e.g.

```python
trained_predictor = EntityClassificationPredictor.from_pretrained(
    model_name_or_path=trainer.config.default_root_dir / "checkpoints",  # one way of getting the path (if you have the trainer)
    entity_name=entity_name,       # specified in EntityClassificationTrainConfig. Probably "tokens".
    context_name=context_name,     # specified in EntityClassificationTrainConfig. Probably "pages".
)

token_tags = trained_predictor.predict(doc=doc)
```