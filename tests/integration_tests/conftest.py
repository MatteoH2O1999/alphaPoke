import gc
import pytest


@pytest.fixture(autouse=True)
def clean_memory():
    gc.collect()
