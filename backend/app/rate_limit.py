"""Rate limiting configuration.

Separated from main.py to avoid circular imports when endpoints
need to apply per-route rate limits.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["100/minute"])
