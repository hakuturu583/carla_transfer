"""Read CARLA's OpenDRIVE with roadgen.

roadgen's reader is built on a strict OpenDRIVE 1.7 model; CARLA's RoadRunner
exported maps (Town01..Town10HD) bend the spec in a few places that the strict
parser rejects outright. :func:`sanitize_carla_xodr` rewrites just those
before the file is handed over. None of the rewrites touch geometry.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

# (pattern, replacement, why)
_FIXES: list[tuple[str, str, str]] = [
    (
        r"<userData>.*?</userData>|<userData\s*/>",
        "",
        "RoadRunner writes <userData><vectorScene/></userData>; the 1.7 model requires a "
        "`code` attribute on userData",
    ),
    (
        r"<roadMark(?![^>]*\bcolor=)([^>]*)/>",
        r'<roadMark\1 color="standard"/>',
        'type="curb" road marks carry no colour; `color` is required (spec default: standard)',
    ),
    (
        r'(<object\b[^>]*\btype=")-1(")',
        r"\1none\2",
        'objects with type="-1" are not a spec enum value',
    ),
    (
        r"<cornerLocal(?![^>]*\bheight=)([^>]*)/>",
        r'<cornerLocal\1 height="0"/>',
        "outline corners omit the required `height`",
    ),
    (
        r"<cornerRoad(?![^>]*\bheight=)([^>]*)/>",
        r'<cornerRoad\1 height="0"/>',
        "outline corners omit the required `height`",
    ),
]


def sanitize_carla_xodr(text: str) -> str:
    for pattern, repl, _why in _FIXES:
        text = re.sub(pattern, repl, text, flags=re.S)
    return text


def read_carla_xodr(path: str | Path):
    """Return a ``roadgen.Map`` for a CARLA .xodr (sanitized copy is read)."""
    import roadgen

    text = Path(path).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / Path(path).name
        clean.write_text(sanitize_carla_xodr(text), encoding="utf-8")
        return roadgen.read_opendrive(str(clean))
