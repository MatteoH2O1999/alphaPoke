#
# A pokémon showdown battle-bot project based on reinforcement learning techniques.
# Copyright (C) 2022 Matteo Dell'Acqua
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Plots the evaluation with the relative 95% confidence interval
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import time


def plot_eval(evaluations, save=False, path="./logs"):
    if save:
        sns.set_theme()
    else:
        sns.set_theme("talk")
    sns.set_palette("colorblind")
    main_color = sns.color_palette()[0]
    colors = sns.color_palette()[1:4]
    evaluations = evaluations[1:]
    baseline_players = [
        "Random player",
        "Max base power player",
        "Simple heuristics player",
    ]
    baseline_values = [1, 7.665994, 128.757145]
    for player, value, color in zip(baseline_players, baseline_values, colors):
        plt.axhline(
            value,
            label=player,
            color=color,
            linestyle="dashed",
            zorder=1,
            linewidth=0.9,
        )
    players = []
    values = []
    errors = []
    max_value = 0
    for evaluation in evaluations:
        players.append(evaluation[0])
        value = evaluation[1][0]
        values.append(value)
        confidence_values = evaluation[1][1]
        if confidence_values[1] > max_value:
            max_value = confidence_values[1]
        errors.append((value - confidence_values[0], confidence_values[1] - value))
    errors = __error_shape(errors)
    plt.errorbar(
        players,
        values,
        errors,
        fmt="none",
        ecolor="black",
        capsize=3,
        elinewidth=1,
        zorder=2,
    )
    plt.scatter(
        players, values, zorder=3, edgecolors="black", linewidths=0.5, color=main_color
    )
    plt.legend()
    plt.xlabel("")
    plt.ylim(0, max_value * 1.025)
    plt.title(
        "Player strength evaluation (with error bars delimiting 95% confidence interval)"
    )
    plt.xticks(rotation=20, horizontalalignment="right")
    plt.ylabel("Player strength")
    if save:
        os.makedirs(path, exist_ok=True)
        current_time = time.localtime()
        filename = (
            f"evaluation {current_time[2]:02}-{current_time[1]:02}-{current_time[0]:04}"
            f" {current_time[3]:02}-{current_time[4]:02}-{current_time[5]:02}.png"
        )
        file_path = os.path.join(path, filename)
        plt.tight_layout()
        plt.savefig(file_path, backend="agg")
    else:
        if max_value < baseline_values[2]:
            plt.ylim(0, baseline_values[2] * 1.025)
        plt.show()


def __error_shape(errors):
    to_return = [[], []]
    for e in errors:
        to_return[0].append(e[0])
        to_return[1].append(e[1])
    return to_return
