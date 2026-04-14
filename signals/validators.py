"""Signal 유효성 검사."""
from __future__ import annotations

from signals.models import Signal


def validate(signal: Signal) -> bool:
    if signal.entry_price <= 0:
        return False
    if signal.direction == "long":
        return signal.tp_price > signal.entry_price > signal.sl_price
    else:
        return signal.tp_price < signal.entry_price < signal.sl_price
