"""Shared visual language for the dashboard and the report figures.

Colours come from a validated palette: the categorical slots are assigned in a
fixed order (never cycled) and the status colours are reserved for reliability
state and never reused as a series colour. Status is always shown with a symbol
and a word next to it, so state never rests on colour alone.
"""
from __future__ import annotations

# categorical slots, in fixed assignment order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MAX_SERIES = len(SERIES)

# reserved status palette
STATUS_COLOR = {
    "Reliable": "#0ca30c",       # good
    "Degraded": "#fab219",       # warning
    "Failure-prone": "#d03b3b",  # critical
}
STATUS_SYMBOL = {
    "Reliable": "●",        # filled circle
    "Degraded": "▲",        # triangle
    "Failure-prone": "✖",   # cross
}
STATUS_ORDER = ["Reliable", "Degraded", "Failure-prone"]

# chart chrome
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

FONT = "system-ui, -apple-system, Segoe UI, sans-serif"


def status_label(status: str) -> str:
    """Symbol + word, so the state never depends on colour alone."""
    return "{} {}".format(STATUS_SYMBOL.get(status, "○"), status)


def altair_theme():
    """Recessive grid and axes, ink-coloured text, no chart-junk."""
    return {
        "config": {
            "background": SURFACE,
            "font": FONT,
            "view": {"stroke": "transparent", "continuousHeight": 260},
            "axis": {
                "labelColor": INK_MUTED,
                "titleColor": INK_SECONDARY,
                "titleFontWeight": 500,
                "labelFontSize": 11,
                "titleFontSize": 11,
                "gridColor": GRIDLINE,
                "gridWidth": 1,
                "domainColor": BASELINE,
                "tickColor": BASELINE,
                "tickSize": 4,
            },
            "legend": {
                "labelColor": INK_SECONDARY,
                "titleColor": INK_SECONDARY,
                "labelFontSize": 11,
                "titleFontSize": 11,
                "symbolType": "stroke",
                "symbolStrokeWidth": 3,
            },
            "title": {
                "color": INK_PRIMARY,
                "fontSize": 13,
                "fontWeight": 600,
                "anchor": "start",
                "offset": 12,
            },
            "line": {"strokeWidth": 2},
            "point": {"size": 64},
            "bar": {"cornerRadiusEnd": 4},
            "rule": {"color": INK_MUTED},
        }
    }
