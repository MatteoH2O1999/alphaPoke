import pytest

from unittest.mock import patch

from cross_eval import main


@pytest.mark.asyncio
async def test_cross_evaluate_integration():
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
