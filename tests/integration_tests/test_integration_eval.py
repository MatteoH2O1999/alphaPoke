import pytest

from unittest.mock import patch

from eval import main


@pytest.mark.asyncio
@pytest.mark.timeout(360)
@pytest.mark.flaky
async def test_eval_integration():
    with patch("sys.argv", ["", "300", "gen8randombattle", "dad"]):
        await main()
