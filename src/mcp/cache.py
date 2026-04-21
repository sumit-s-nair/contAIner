"""In-memory LRU cache with per-entry TTL for the MCP doc server.

Cache key is a (tool, operation, package, os) tuple.
Two TTL tiers:
  • Registry data  — 1 hour   (3 600 s)
  • Docs pages     — 24 hours (86 400 s)
Max 500 entries; LRU eviction when full.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

# Default TTL constants (seconds)
REGISTRY_TTL: int = 3_600       # 1 hour
DOCS_TTL: int = 86_400          # 24 hours
MAX_CACHE_SIZE: int = 500


@dataclass(frozen=True)
class _CacheEntry:
    """Single cache entry with its value and expiry timestamp."""
    value: Any
    expires_at: float


class TTLCache:
    """Thread-safe LRU cache with per-entry TTL.

    Usage::

        cache = TTLCache(max_size=500)
        cache.set(("pip", "install", "numpy", "linux"), doc_chunk, ttl=3600)
        hit = cache.get(("pip", "install", "numpy", "linux"))
    """

    def __init__(self, max_size: int = MAX_CACHE_SIZE) -> None:
        self._max_size = max_size
        self._store: OrderedDict[tuple, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────

    def get(self, key: tuple) -> Any | None:
        """Return cached value or ``None`` if missing / expired."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                # Expired — remove and return miss
                del self._store[key]
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            return entry.value

    def set(self, key: tuple, value: Any, ttl: int = REGISTRY_TTL) -> None:
        """Insert or update an entry.  Evicts LRU if at capacity."""
        expires_at = time.monotonic() + ttl
        with self._lock:
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self._max_size:
                # Evict the oldest (least recently used)
                self._store.popitem(last=False)
            self._store[key] = _CacheEntry(value=value, expires_at=expires_at)

    def invalidate(self, key: tuple) -> bool:
        """Remove a single entry.  Returns True if it existed."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        """Drop every entry."""
        with self._lock:
            self._store.clear()

    def prune_expired(self) -> int:
        """Remove all expired entries.  Returns number removed."""
        now = time.monotonic()
        removed = 0
        with self._lock:
            expired_keys = [k for k, v in self._store.items() if now > v.expires_at]
            for k in expired_keys:
                del self._store[k]
                removed += 1
        return removed

    @property
    def size(self) -> int:
        return len(self._store)

    def __contains__(self, key: tuple) -> bool:
        return self.get(key) is not None

    def __repr__(self) -> str:
        return f"TTLCache(size={self.size}, max={self._max_size})"
