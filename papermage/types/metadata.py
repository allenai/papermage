"""


@lucas

"""

from copy import deepcopy
from dataclasses import MISSING, Field, fields, is_dataclass
from functools import wraps
import inspect
import logging

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

__all__ = ["store_field_in_metadata", "Metadata"]


class _DEFAULT:
    # object to keep track if a default value is provided when getting
    # or popping a key from Metadata
    ...


class Metadata:
    """An object that contains metadata for an annotation.
    It supports dot access and dict-like access."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.set(k, v)

    @overload
    def get(self, key: str) -> Any:
        """Get value with name `key` in metadata"""
        ...

    @overload
    def get(self, key: str, default: Any) -> Any:
        """Get value with name `key` in metadata"""
        ...

    def get(self, key: str, default: Optional[Any] = _DEFAULT) -> Any:
        """Get value with name `key` in metadata;
        if not found, return `default` if specified,
        otherwise raise `None`"""
        if key in self.__dict__:
            return self.__dict__[key]
        elif default != _DEFAULT:
            return default
        else:
            logging.warning(f"{key} not found in metadata")
            return None

    def has(self, key: str) -> bool:
        """Check if metadata contains key `key`; return `True` if so,
        `False` otherwise"""
        return key in self.__dict__

    def set(self, key: str, value: Any) -> None:
        """Set `key` in metadata to `value`; key must be a valid Python
        identifier (that is, a valid variable name) otherwise,
        raise a ValueError"""
        if not key.isidentifier():
            raise ValueError(
                f"`{key}` is not a valid variable name, "
                "so it cannot be used as key in metadata"
            )
        self.__dict__[key] = value

    @overload
    def pop(self, key: str) -> Any:
        """Remove & returns value for `key` from metadata;
        raise `KeyError` if not found"""
        ...

    @overload
    def pop(self, key: str, default: Any) -> Any:
        """Remove & returns value for `key` from metadata;
        if not found, return `default`"""
        ...

    def pop(self, key: str, default: Optional[Any] = _DEFAULT) -> Any:
        """Remove & returns value for `key` from metadata;
        if not found, return `default` if specified,
        otherwise raise `KeyError`"""
        if key in self.__dict__:
            return self.__dict__.pop(key)
        elif default != _DEFAULT:
            return default
        else:
            raise KeyError(f"{key} not found in metadata")

    def keys(self) -> Iterable[str]:
        """Return an iterator over the keys in metadata"""
        return self.__dict__.keys()

    def values(self) -> Iterable[Any]:
        """Return an iterator over the values in metadata"""
        return self.__dict__.values()

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Return an iterator over <key, value> pairs in metadata"""
        return self.__dict__.items()

    # Interfaces from loading/saving to dictionary
    def to_json(self) -> Dict[str, Any]:
        """Return a dict representation of metadata"""
        return deepcopy(self.__dict__)

    @classmethod
    def from_json(cls, di: Dict[str, Any]) -> "Metadata":
        """Create a Metadata object from a dict representation"""
        metadata = cls()
        for k, v in di.items():
            metadata.set(k, v)
        return metadata

    # The following methods are to ensure equality between metadata
    # with same keys and values
    def __len__(self) -> int:
        return len(self.__dict__)

    def __eq__(self, __o: object) -> bool:
        if not isinstance(__o, Metadata):
            return False
        if len(self) != len(__o):
            return False
        for k in __o.keys():
            if k not in self.keys() or self[k] != __o[k]:
                return False
        return True

    # The following methods are for compatibility with the dict interface
    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __iter__(self) -> Iterable[str]:
        return self.keys()

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        return self.set(key, value)

    # The following methods are for compatibility with the dot access interface
    def __getattr__(self, key: str) -> Any:
        return self.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        return self.set(key, value)

    def __delattr__(self, key: str) -> Any:
        return self.pop(key)

    # The following methods return a nice representation of the metadata
    def __repr__(self) -> str:
        return f"Metadata({repr(self.__dict__)})"

    def __str__(self) -> str:
        return f"Metadata({str(self.__dict__)})"

    # Finally, we need to support pickling/copying
    def __deepcopy__(self, memo: Dict[int, Any]) -> "Metadata":
        return Metadata.from_json(deepcopy(self.__dict__, memo))


T = TypeVar("T")


def store_field_in_metadata(
    field_name: str,
    getter_fn: Optional[Callable[[T], Any]] = None,
    setter_fn: Optional[Callable[[T, Any], None]] = None,
) -> Callable[[Type[T]], Type[T]]:
    """This decorator is used to store a field that was previously part of
    a dataclass in a metadata attribute, while keeping it accessible as an
    attribute using the .field_name notation. Example:

    @store_field_in_metadata('field_a')
    @dataclass
    class MyDataclass:
        metadata: Metadata = field(default_factory=Metadata)
        field_a: int = 3
        field_b: int = 4

    d = MyDataclass()
    print(d.field_a) # 3
    print(d.metadata)   # Metadata({'field_a': 3})
    print(d) # MyDataclass(field_a=3, field_b=4, metadata={'field_a': 3})

    d = MyDataclass(field_a=5)
    print(d.field_a) # 5
    print(d.metadata)   # Metadata({'field_a': 5})
    print(d) # MyDataclass(field_a=5, field_b=4, metadata={'field_a': 5})

    Args:
        field_name: The name of the field to store in the metadata.
        getter_fn: A function that takes the dataclass instance and
            returns the value of the field. If None, the field's value is
            looked up from the metadata dictionary.
        setter_fn: A function that takes the dataclass instance and a value
            for the field and sets it. If None, the field's value is added
            to the metadata dictionary.
    """

    def wrapper_fn(
        cls_: Type[T],
        wrapper_field_name: str = field_name,
        wrapper_getter_fn: Optional[Callable[[T], Any]] = getter_fn,
        wrapper_setter_fn: Optional[Callable[[T, Any], None]] = setter_fn,
    ) -> Type[T]:
        """
        This wrapper consists of three steps:
        1. Basic checks to determine if a field can be stored in the metadata.
           This includes checking that cls_ is a dataclass, that cls_ has a
           metadata attribute, and that the field is a field of cls_.
        2. Wrap the init method of cls_ to ensure that, if a field is specified
           in the metadata, it is *NOT* overwritten by the default value of the
           field. (keep reading through this function for more details)
        3. Create getter and setter methods for the field; these are going to
           override the original attribute and will be responsible for querying
           the metadata for the value of the field.

        Args:
            cls_ (Type[T]): The dataclass to wrap.
            wrapper_field_name (str): The name of the field to store in the
                metadata.
            wrapper_getter_fn (Optional[Callable[[T], Any]]): A function that
                takes returns the value of the field. If None, the getter
                is a simple lookup in the metadata dictionary.
            wrapper_setter_fn (Optional[Callable[[T, Any], None]]): A function
                that is used to set the value of the field. If None, the setter
                is a simple addition to the metadata dictionary.
        """

        # # # # # # # # # # # # STEP 1: BASIC CHECKS # # # # # # # # # # # # #
        if not (is_dataclass(cls_)):
            raise TypeError("add_deprecated_field only works on dataclasses")

        dataclass_fields = {field.name: field for field in fields(cls_)}

        # ensures we have a metadata dict where to store the field value
        if "metadata" not in dataclass_fields:
            raise TypeError(
                "add_deprecated_field requires a `metadata` field"
                "in the dataclass of type dict."
            )

        if not issubclass(dataclass_fields["metadata"].type, Metadata):
            raise TypeError(
                "add_deprecated_field requires a `metadata` field "
                "in the dataclass of type Metadata, not "
                f"{dataclass_fields['metadata'].type}."
            )

        # ensure the field is declared in the dataclass
        if wrapper_field_name not in dataclass_fields:
            raise TypeError(
                f"add_deprecated_field requires a `{wrapper_field_name}` field"
                "in the dataclass."
            )
        # # # # # # # # # # # # # # END OF STEP 1 # # # # # # # # # # # # # # #

        # # # # # # # # # # # # # STEP 2: WRAP INIT # # # # # # # # # # # # # #
        # In the following comment, we explain the need for step 2.
        #
        # We want to make sure that if a field is specified in the metadata,
        # the default value of the field provided during class annotation does
        # not override it. For example, consider the following code:
        #
        #   @store_field_in_metadata('field_a')
        #   @dataclass
        #   class MyDataclass:
        #       metadata: Metadata = field(default_factory=Metadata)
        #       field_a: int = 3
        #
        # If we don't disable wrap the __init__ method, the following code
        # will print `3`
        #
        #   d = MyDataclass(metadata={'field_a': 5})
        #   print(d.field_a)
        #
        # but if we do, it will work fine and print `5` as expected.
        #
        # The reason why this occurs is that the __init__ method generated
        # by a dataclass uses the default value of the field to initialize the
        # class if a default is not provided.
        #
        # Our solution is rather simple: before calling the dataclass init,
        # we look if:
        #   1. A `metadata` argument is provided in the constructor, and
        #   2. The `metadata` argument contains a field with name ``
        #
        # To disable the auto-init, we have to do two things:
        #   1. create a new dataclass that inherits from the original one,
        #      but with init=False for field wrapper_field_name
        #   2. create a wrapper for the __init__ method of the new dataclass
        #      that, when called, calls the original __init__ method and then
        #      adds the field value to the metadata dict.

        # This signature is going to be used to bind to the args/kwargs during
        # init, which allows easy lookup of arguments/keywords arguments by
        # name.
        cls_signature = inspect.signature(cls_.__init__)

        # We need to save the init method since we will override it.
        cls_init_fn = cls_.__init__

        @wraps(cls_init_fn)
        def init_wrapper(self, *args, **kwargs):
            # parse the arguments and keywords arguments
            arguments = cls_signature.bind(self, *args, **kwargs).arguments

            # this adds the metadata to kwargs if it is not already there
            metadata = arguments.setdefault("metadata", Metadata())

            # this is the main check:
            # (a) the metadata argument contains the field we are storing in
            # the metadata, and (b) the field is not in args/kwargs, then we
            # pass the field value in the metadata to the original init method
            # to prevent it from being overwritten by its default value.
            if (
                wrapper_field_name in metadata
                and wrapper_field_name not in arguments
            ):
                arguments[wrapper_field_name] = metadata[wrapper_field_name]

            # type: ignore is due to pylance not recognizing that the
            # arguments in the signature contain a `self` key
            cls_init_fn(**arguments)  # type: ignore

        setattr(cls_, "__init__", init_wrapper)
        # # # # # # # # # # # # # # END OF STEP 2 # # # # # # # # # # # # # # #

        # # # # # # # # # # # STEP 3: GETTERS & SETTERS # # # # # # # # # # # #

        # We add the getter from here on:
        if wrapper_getter_fn is None:
            # create property for the deprecated field, as well as a setter
            # that will add to the underlying metadata dict
            def _wrapper_getter_fn(
                self, field_spec: Field = dataclass_fields[wrapper_field_name]
            ):
                # we expect metadata to be of type Metadata
                metadata: Union[Metadata, None] = getattr(
                    self, "metadata", None
                )
                if metadata is not None and wrapper_field_name in metadata:
                    return metadata.get(wrapper_field_name)
                elif field_spec.default is not MISSING:
                    return field_spec.default
                elif field_spec.default_factory is not MISSING:
                    return field_spec.default_factory()
                else:
                    raise AttributeError(
                        f"Value for attribute '{wrapper_field_name}' "
                        "has not been set."
                    )

            # this avoids mypy error about redefining an argument
            wrapper_getter_fn = _wrapper_getter_fn

        field_property = property(wrapper_getter_fn)

        # We add the setter from here on:
        if wrapper_setter_fn is None:

            def _wrapper_setter_fn(self: T, value: Any) -> None:
                # need to use getattr otherwise pylance complains
                # about not knowing if 'metadata' is available as an
                # attribute of self (which it is, since we checked above
                # that it is in the dataclass fields)
                metadata: Union[Metadata, None] = getattr(
                    self, "metadata", None
                )

                if metadata is None:
                    raise RuntimeError(
                        "all deprecated fields must be declared after the "
                        f"`metadata` field; however, `{wrapper_field_name}`"
                        " was declared before. Fix your class definition."
                    )

                metadata.set(wrapper_field_name, value)

            # this avoids mypy error about redefining an argument
            wrapper_setter_fn = _wrapper_setter_fn

        # make a setter for the deprecated field
        field_property = field_property.setter(wrapper_setter_fn)

        # assign the property to the dataclass
        setattr(cls_, wrapper_field_name, field_property)
        # # # # # # # # # # # # # # END OF STEP 3 # # # # # # # # # # # # # # #

        return cls_

    return wrapper_fn
