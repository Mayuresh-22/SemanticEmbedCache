from typing import Any
from src.storage.BaseStorage import BaseStorage


class InMemStorage(BaseStorage):
    def __init__(self):
        self.storage: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.storage.get(key)

    def get_all_keys(self) -> list[str]:
        return list(self.storage.keys())

    def set(self, key: str, value: Any) -> None:
        self.storage[key] = value
