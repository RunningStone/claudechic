"""Lightweight profiling utilities for performance analysis."""

import atexit
import functools
import inspect
import time
from collections import defaultdict

_stats: dict[str, dict] = defaultdict(lambda: {"count": 0, "total": 0.0, "max": 0.0})


def profile(fn):
    """Decorator to track function call count and timing."""
    label = fn.__qualname__

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _stats[label]["count"] += 1
        _stats[label]["total"] += elapsed
        _stats[label]["max"] = max(_stats[label]["max"], elapsed)
        return result

    @functools.wraps(fn)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _stats[label]["count"] += 1
        _stats[label]["total"] += elapsed
        _stats[label]["max"] = max(_stats[label]["max"], elapsed)
        return result

    if inspect.iscoroutinefunction(fn):
        return async_wrapper
    return wrapper


def print_stats():
    """Print collected statistics."""
    if not _stats:
        return
    print("\n" + "=" * 80)
    print("PROFILING STATISTICS")
    print("=" * 80)
    print(f"\n{'Function':<45} {'Calls':>8} {'Total':>10} {'Avg':>10} {'Max':>10}")
    print("-" * 80)
    for name, data in sorted(_stats.items(), key=lambda x: -x[1]["total"]):
        avg = data["total"] / data["count"] * 1000 if data["count"] else 0
        print(f"{name:<45} {data['count']:>8} {data['total']*1000:>9.1f}ms {avg:>9.2f}ms {data['max']*1000:>9.2f}ms")
    print("=" * 80 + "\n")


atexit.register(print_stats)
