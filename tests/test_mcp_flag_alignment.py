"""Regression: compression must keep each flag paired with its own description.

Before the fix, reassemble._inject_compressed_text joined every flag
description with '\\n', compressed the blob, then split it on '\\n' and
re-assigned descriptions by position.  The extractive compressor re-joins kept
sentences with spaces, so line boundaries vanished and later flags received
other flags' text (snapshot: '--no-deps' described as '--target').
"""

from src.mcp.compress import compress_chunk
from src.mcp.models import DocChunk


def _chunk():
    flags = [
        {"flag": "-r, --requirement <file>",
         "description": "Install from the given requirements file. This option can be used multiple times."},
        {"flag": "--no-deps",
         "description": "Don't install package dependencies."},
        {"flag": "--only-binary <format_control>",
         "description": "Do not use source packages. Can be supplied multiple times, and each time adds to "
                        "the existing value. Accepts either :all: to disable all source packages."},
        {"flag": "-t, --target <dir>",
         "description": "Install packages into <dir>. By default this will not replace existing files."},
        {"flag": "--no-build-isolation",
         "description": "Disable isolation when building a modern source distribution. Build dependencies "
                        "specified by PEP 518 must be already installed if this option is used."},
    ]
    return DocChunk(
        tool="pip", operation="install",
        command_syntax="python -m pip install [options] <requirement specifier>",
        key_flags=flags, examples=["python -m pip install -r requirements.txt"],
        source_urls=[], os_specific_notes="Use a virtual environment. The weather is nice today.",
    )


import re

import pytest


@pytest.mark.parametrize("notes", ["", "Use a virtual environment. The weather is nice today."])
def test_each_flag_keeps_its_own_description(notes):
    orig = _chunk()
    orig.os_specific_notes = notes
    out = compress_chunk(orig, method="extractive", cache=None)
    assert [f["flag"] for f in out.key_flags] == [f["flag"] for f in orig.key_flags]
    for before, after in zip(orig.key_flags, out.key_flags):
        first_sentence = before["description"].split(". ")[0].rstrip(".")
        assert after["description"].startswith(first_sentence), (before["flag"], after["description"])
        # and every kept sentence comes from THIS flag's own description
        for sent in re.split(r"(?<=\.)\s+", after["description"].strip()):
            assert sent.rstrip(".") in before["description"], (before["flag"], sent)


def test_notes_do_not_absorb_flag_text():
    orig = _chunk()
    out = compress_chunk(orig, method="extractive", cache=None)
    assert "By default this will not replace" not in out.os_specific_notes
    assert out.os_specific_notes.startswith("Use a virtual environment")


def test_no_description_is_blanked():
    out = compress_chunk(_chunk(), method="extractive", cache=None)
    assert all(f["description"].strip() for f in out.key_flags)


def test_input_chunk_not_mutated():
    orig = _chunk()
    snapshot = [dict(f) for f in orig.key_flags]
    compress_chunk(orig, method="extractive", cache=None)
    assert orig.key_flags == snapshot
