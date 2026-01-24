from abc import ABC, abstractmethod
from typing import Any


class BaseStorage(ABC):
    @abstractmethod
    def get(self, key: str) -> Any:
        raise NotImplementedError("The 'get' method is not implemented.")

    @abstractmethod
    def get_all_keys(self) -> list[str]:
        raise NotImplementedError("The 'get_all_keys' method is not implemented.")

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        raise NotImplementedError("The 'set' method is not implemented.")
