import pytest

from unittest.mock import patch

from rank import main
from utils.invalid_argument import InvalidArgumentNumber


@pytest.mark.asyncio
async def test_rank_wrong_args():
    with patch("sys.argv", ["test", "test", "test"]):
        with pytest.raises(InvalidArgumentNumber):
            await main()
