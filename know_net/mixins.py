import abc
import asyncio
import os
from typing import (
    Any,
    AsyncGenerator,
    Generic,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
)

import diskcache

T = TypeVar("T", covariant=True)


def iterate_through(generator: AsyncGenerator[T, None]) -> List[T]:
    """
    synchronous iteration over an asynchronous generator
    """
    return asyncio.run(iterate_through_async_generator(generator))


async def iterate_through_async_generator(
    async_generator: AsyncGenerator[T, None]
) -> List[T]:
    loaded_stuff: List[T] = list()
    async for article in async_generator:
        loaded_stuff.append(article)
    return loaded_stuff


###
# Cache
###

CACHE_DIR = ".knownet_cache/"


class WithDiskCache(abc.ABC):
    _cache: Optional[diskcache.Cache] = None

    def from_cache(self, key: Any) -> Optional[object]:
        return self.cache.get(key)

    def set_in_cache(self, key: Any, value: object) -> None:
        self.cache.set(key, value)

    def is_in_cache(self, key: Any) -> bool:
        return key in self.cache

    def clear_cache(self) -> None:
        self.cache.clear()

    @property
    @abc.abstractmethod
    def cache_name(self) -> str:
        ...

    @property
    def cache(self) -> diskcache.Cache:
        if self._cache is None:
            return self._create_cache()
        else:
            return self._cache

    @property
    def cache_path(self) -> Union[str, os.PathLike]:
        return os.path.join(CACHE_DIR, self.cache_name)

    def _create_cache(self) -> diskcache.Cache:
        self._cache = diskcache.Cache(self.cache_path)
        return self._cache
