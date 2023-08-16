from collections import UserList
from typing import List, Union, Any

from .entity import Entity

class Layer(UserList):
    """Wraps a list of entities"""
    def __init__(self, entities: List[Entity] = None):
        if entities is None:
            super().__init__()
        else:
            super().__init__(entities)
    
    def __getitem__(self, index: Union[int, slice]) -> Union[Entity, "Layer"]:
        if isinstance(index, int):
            return self.data[index]
        else:
            return Layer(self.data[index])
    
    def __getattr__(self, field: str) -> "Layer":
        return Layer([
            getattr(entity, field) for entity in self.data
        ])
    
