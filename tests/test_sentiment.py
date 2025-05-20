import pytest
from src.analysis_functions import get_sentiment

def test_get_sentiment():
    assert get_sentiment("Very good.")['Sentiment'] == "Positive"
    assert get_sentiment("Very bad.")['Sentiment'] == "Negative"
