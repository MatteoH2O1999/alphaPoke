# Plots the evaluation with the relative 95% confidence interval
import matplotlib.pyplot as plt
import os.path
import seaborn as sns
import time


def plot_eval(evaluations, save=False, path='./plot'):
    sns.set_theme()
    sns.set_palette('colorblind')
    evaluations = evaluations[1:]
    baseline_players = ['RandomPlayer', 'MaxBasePowerPlayer', 'SimpleHeuristicsPlayer']
    baseline_values = [1, 7.665994, 128.757145]
    tmp = []
    for player, value in zip(baseline_players, baseline_values):
        tmp.append([player, (value, (value, value))])
    for e in evaluations:
        tmp.append(e)
    evaluations = tmp
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
    plt.errorbar(players, values, errors, fmt='none', ecolor='black', capsize=3, elinewidth=1)
    plt.scatter(players, values, zorder=2, edgecolors='black', linewidths=0.5)
    plt.xlabel('')
    plt.ylim(0, max_value * 1.1)
    plt.title('Player strength evaluation (with error bars delimiting 95% confidence interval)')
    plt.xticks(rotation=20, horizontalalignment='right')
    plt.ylabel('Player strength')
    if save:
        os.makedirs(path, exist_ok=True)
        current_time = time.localtime()
        filename = f'plot {current_time[2]:02}-{current_time[1]:02}-{current_time[0]:04}.png'
        file_path = os.path.join(path, filename)
        plt.savefig(file_path, backend='agg', bbox_inches='tight')
    else:
        plt.show()


def __error_shape(errors):
    to_return = [[], []]
    for e in errors:
        to_return[0].append(e[0])
        to_return[1].append(e[1])
    return to_return
