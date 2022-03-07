import pytest

from unittest.mock import patch, MagicMock

from agents.dad import Dad
from agents.eight_year_old_me import EightYearOldMe
from agents.twenty_year_old_me import TwentyYearOldMe
from cross_eval import main
from utils.invalid_argument import InvalidArgumentNumber, InvalidArgument


@pytest.mark.asyncio
async def test_cross_evaluate_wrong_args():
    with patch("sys.argv", ["test", "test"]):
        with pytest.raises(InvalidArgumentNumber):
            await main()
    with patch("sys.argv", ["test", "test", "test"]):
        with pytest.raises(InvalidArgument):
            await main()
    with patch("sys.argv", ["test", "200", "test", "200", "test"]):
        with pytest.raises(InvalidArgument):
            await main()


class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


@pytest.mark.asyncio
async def test_cross_evaluate_argparse():
    with patch(
        "sys.argv",
        [
            "",
            "2",
            "gen8randombattle",
            "dad",
            "1",
            "8-year-old-me",
            "1",
            "20-year-old-me",
            "1",
        ],
    ), patch("cross_eval.cross_evaluate", new_callable=AsyncMock) as evaluate_mock:
        evaluate_mock.return_value = {}
        await main()
        args, kwargs = evaluate_mock.call_args
        evaluate_mock.assert_called_once()
        assert len(args[0]) == 3
        assert isinstance(args[0][0], Dad)
        assert isinstance(args[0][1], EightYearOldMe)
        assert isinstance(args[0][2], TwentyYearOldMe)
        assert kwargs["n_challenges"] == 2
