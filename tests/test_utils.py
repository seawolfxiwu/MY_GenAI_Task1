import pytest
from src.utils import clean_text

def test_clean_text():
    # Test text shorter than max length
    assert clean_text("hello") == "hello"

    # Test text longer than max length
    long_text = "a" * 200
    assert clean_text(long_text) == "a" * 128 + "..."

    # Test edge case
    assert clean_text("") == ""