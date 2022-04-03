import seaborn as sns
import pytest
from unittest.mock import patch

from utils.plot_eval import plot_eval, __error_shape


def test_error_shape():
    with pytest.raises(TypeError):
        __error_shape([1])
    with pytest.raises(IndexError):
        __error_shape([(1,)])
    errors = [(1, 2), (3.5, 4.6), (8, 6), (3.4, 4.3)]
    new_errors = __error_shape(errors)
    assert len(new_errors) == 2
    assert len(new_errors[0]) == len(errors)
    assert len(new_errors[1]) == len(errors)
    for i, error in enumerate(errors):
        assert new_errors[0][i] == error[0]
        assert new_errors[1][i] == error[1]


def test_plot_eval_no_save():
    with patch("matplotlib.pyplot.errorbar") as mock_errorbar, patch(
        "matplotlib.pyplot.scatter"
    ) as mock_scatter, patch("matplotlib.pyplot.show") as mock_show:
        evaluations = [
            ("players", "evaluations"),
            ("p1", (1.0, (0.5, 1.5))),
            ("p2", (10.0, (5.0, 15.0))),
            ("p3", (100.0, (50.0, 150.0))),
        ]
        plot_eval(evaluations, save=False, path="./plots")
        mock_errorbar.assert_called_with(
            ["p1", "p2", "p3"],
            [1.0, 10.0, 100.0],
            [[0.5, 5.0, 50.0], [0.5, 5.0, 50.0]],
            fmt="none",
            ecolor="black",
            capsize=3,
            elinewidth=1,
            zorder=2,
        )
        mock_scatter.assert_called_with(
            ["p1", "p2", "p3"],
            [1.0, 10.0, 100.0],
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
            color=sns.color_palette("colorblind")[0],
        )
        mock_show.assert_called_once()


def test_plot_eval_save():
    with patch("matplotlib.pyplot.errorbar") as mock_errorbar, patch(
        "matplotlib.pyplot.scatter"
    ) as mock_scatter, patch("matplotlib.pyplot.savefig") as mock_savefig, patch(
        "os.makedirs"
    ) as mock_makedirs, patch(
        "matplotlib.pyplot.tight_layout"
    ) as mock_tight:
        evaluations = [
            ("players", "evaluations"),
            ("p1", (1.0, (0.5, 1.5))),
            ("p2", (10.0, (5.0, 15.0))),
            ("p3", (100.0, (50.0, 150.0))),
        ]
        plot_eval(evaluations, save=True, path="./plots")
        mock_errorbar.assert_called_with(
            ["p1", "p2", "p3"],
            [1.0, 10.0, 100.0],
            [[0.5, 5.0, 50.0], [0.5, 5.0, 50.0]],
            fmt="none",
            ecolor="black",
            capsize=3,
            elinewidth=1,
            zorder=2,
        )
        mock_scatter.assert_called_with(
            ["p1", "p2", "p3"],
            [1.0, 10.0, 100.0],
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
            color=sns.color_palette("colorblind")[0],
        )
        mock_savefig.assert_called_once()
        mock_makedirs.assert_called_once_with("./plots", exist_ok=True)
        mock_tight.assert_called_once()
        args, kwargs = mock_savefig.call_args
        assert (
            "./plots/evaluation" in args[0] or "./plots\\evaluation" in args[0]
        ) and ".png" in args[0]
        assert kwargs["backend"] == "agg"
