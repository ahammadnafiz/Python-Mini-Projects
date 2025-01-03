# tests/test_converter.py
import pytest
from repoprompter.repoprompter.converter import convert_to_text

def test_convert_to_text():
    contents = []  # Mock contents
    text_content = convert_to_text(contents, None)
    assert isinstance(text_content, list)
