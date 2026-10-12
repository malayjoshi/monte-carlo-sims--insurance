import argparse

import pytest

from benchmarks.benchmark_chunks import parse_chunks


def test_parse_chunks():
    assert parse_chunks("1024x128,2048x256") == [(1024, 128), (2048, 256)]


@pytest.mark.parametrize("value", ["bad", "0x128", "1x-2"])
def test_parse_chunks_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        parse_chunks(value)
