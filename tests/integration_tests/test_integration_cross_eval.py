import pytest

from unittest.mock import patch

from cross_eval import main


@pytest.mark.asyncio
@pytest.mark.timeout(90)
@pytest.mark.flaky(max_runs=10, min_passes=1)
async def test_cross_eval_integration():
    with patch(
        "sys.argv",
        [
            "",
            "100",
            "gen8randombattle",
            "dad",
            "1",
            "8-year-old-me",
            "1",
            "20-year-old-me",
            "1",
        ],
    ):
        await main()
