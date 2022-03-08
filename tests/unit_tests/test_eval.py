import pytest

from unittest.mock import patch

from eval import main
from utils.invalid_argument import InvalidArgument


@pytest.mark.asyncio
async def test_eval_invalid_argument():
    with patch("sys.argv", ["", "test", "gen8randombattle"]):
        with pytest.raises(InvalidArgument):
            await main()
