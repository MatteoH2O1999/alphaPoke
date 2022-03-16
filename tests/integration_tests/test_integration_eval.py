import pytest

from unittest.mock import patch

from eval import main


@pytest.mark.asyncio
@pytest.mark.timeout(90)
@pytest.mark.flaky(max_runs=10, min_passes=1)
async def test_eval_integration():
    with patch("sys.argv", ["", "300", "gen8randombattle", "dad"]):
        await main()
