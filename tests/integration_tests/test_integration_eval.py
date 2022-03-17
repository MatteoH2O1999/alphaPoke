import matplotlib.pyplot as plt
import pytest

from unittest.mock import patch

from eval import main


@pytest.mark.asyncio
@pytest.mark.timeout(90)
@pytest.mark.flaky(max_runs=10, min_passes=1)
async def test_eval_integration():
    with patch("sys.argv", ["", "300", "gen8randombattle", "dad"]), patch(
        "matplotlib.pyplot.savefig"
    ) as mock_savefig, patch("os.makedirs") as mock_makedirs:
        plt.switch_backend("agg")
        await main()
        mock_savefig.assert_called_once()
        mock_makedirs.assert_called_once_with("./logs", exist_ok=True)
