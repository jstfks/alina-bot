"""
persona/ — Пакет персонажа Алины.

Публичный API:
  from persona import CORE_PROMPT, build_context_layers

Остальное — внутренняя структура, не импортировать напрямую.
"""

from persona.core import CORE_PROMPT
from persona.layers import build_context_layers

__all__ = ["CORE_PROMPT", "build_context_layers"]
