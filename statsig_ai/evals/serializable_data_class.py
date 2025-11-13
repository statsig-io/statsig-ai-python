from dataclasses import fields, is_dataclass
from typing import Any, TypeVar, cast
from datetime import datetime, date
from enum import Enum

T = TypeVar("T", bound="SerializableDataClass")


class SerializableDataClass:
    """
    Base class for dataclasses that provides easy serialization

    Usage:
        @dataclass
        class MyClass(SerializableDataclass):
            name: str
            value: int
    """

    def to_dict(self) -> dict[str, Any]:
        """Convert dataclass instance to dictionary, handling nested structures. Exclude None values"""
        result = {}
        for field in fields(cast(Any, self)):
            value = getattr(self, field.name)
            serialized_value = self._serialize_value(value)
            if serialized_value is not None:
                result[field.name] = serialized_value
        return result

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        """Serialize a single value, handling various types."""
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if is_dataclass(value) and hasattr(value, "to_dict"):
            return value.to_dict()  # type: ignore
        if isinstance(value, dict):
            return {k: SerializableDataClass._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [SerializableDataClass._serialize_value(item) for item in value]
        # Fallback for other types
        return str(value)
