# tests/test_structurer.py
import pytest
from repoprompter.repoprompter.structurer import structure_text_for_llm

def test_structure_text_for_llm():
    text_content = ["File: example.py\nprint('Hello, World!')\n"]
    structured_text = structure_text_for_llm(text_content)
    assert isinstance(structured_text, str)
