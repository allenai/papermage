import re
from typing import Dict, List, Optional, Tuple, Union

from papermage.magelib import Entity, Prediction


def group_by(
    entities: List[Entity], metadata_field: str, metadata_values_map: Optional[Dict[str, str]] = None
) -> Tuple[Prediction, ...]:
    """Group entities by the value of a field in metadata. After grouping,
    entities are annotated onto the document object.

    Args:
        doc (Document): the document object
        entities (List[Entity]): the entities to group
        metadata_field (str): the field name to group by
        metadata_values (Optional[List[str]], optional): a map of names to use for the groups.
            If not provided, the metadata values will be used as the group names. Defaults to None.

    Returns:
        Document: the document object with the grouped entities
    """
    metadata_values_map = metadata_values_map or {}

    groups: Dict[str, List[Entity]] = {k: [] for k in metadata_values_map.values()}
    for entity in entities:
        mt_val = entity.metadata[metadata_field]  # type: ignore
        mt_val = metadata_values_map.get(mt_val, mt_val)

        # check if the a valid name for a layer
        if not isinstance(mt_val, str):
            raise ValueError(f"Expected metadata value to be a string, but got {type(mt_val)}: {mt_val}")
        elif re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", mt_val) is None:
            raise ValueError(f"Expected metadata value to be a valid Python identifier, but got: {mt_val}")

        new_entity = Entity(
            boxes=getattr(entity, "boxes", None),
            spans=getattr(entity, "spans", None),
            metadata=getattr(entity, "metadata", None),
        )
        groups.setdefault(mt_val, []).append(new_entity)

    return tuple(
        Prediction(name=group_name, entities=group_entities) for group_name, group_entities in groups.items()
    )
