"""Plotting module."""

import matplotlib.pyplot as plt
import numpy as np

def adjust_plot(plt, axes):
    # Turn off top and right axes
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    # Adjust spacing between subplots
    plt.tight_layout()


def plot_membership_functions(code_tuple, report_tuple, punctuality_tuple, grade_tuple):

        # Get values from tuples
        code_domain, poor_code_quality, avg_code_quality, high_code_quality = code_tuple
        report_domain, poor_report_quality, avg_report_quality, high_report_quality = report_tuple
        punctuality_domain, poor_punctuality, avg_punctuality, high_punctuality = punctuality_tuple
        grade_domain, grade_wo, grade_bd, grade_md, grade_gd, grade_ex = grade_tuple

        # Create a figure and a set of subplots
        fig_size = 12
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(fig_size, fig_size))
        axes = axes.flatten()

        # Self-assessment
        axes[0].plot(code_domain, poor_code_quality, 'C0')
        axes[0].plot(code_domain, avg_code_quality, 'C1')
        axes[0].plot(code_domain, high_code_quality, 'C2')
        axes[0].set_title('Code Quality')
        axes[0].legend(['Poor', 'Medium', 'High'], loc='center left')

        # Homework
        axes[1].plot(report_domain, poor_report_quality, 'C0')
        axes[1].plot(report_domain, avg_report_quality, 'C1')
        axes[1].plot(report_domain, high_report_quality, 'C2')
        axes[1].set_title('Report Quality')
        axes[1].legend(['Poor', 'Medium', 'High'], loc='center left')

        # Stress
        axes[2].plot(punctuality_domain, poor_punctuality, 'C0')
        axes[2].plot(punctuality_domain, avg_punctuality, 'C1')
        axes[2].plot(punctuality_domain, high_punctuality, 'C2')
        axes[2].set_title('Punctuality')
        axes[2].legend(['Low', 'Medium', 'High'], loc='center left')

        # Grade
        axes[3].plot(grade_domain, grade_wo, 'C0')
        axes[3].plot(grade_domain, grade_bd, 'C1')
        axes[3].plot(grade_domain, grade_md, 'C2')
        axes[3].plot(grade_domain, grade_gd, 'C3')
        axes[3].plot(grade_domain, grade_ex, 'C4')
        axes[3].set_title('Grade')
        axes[3].legend(['Worst', 'Bad', 'Mediocre', 'Good', 'Excellent'], loc='center left')

        adjust_plot(plt, axes)


def plot_active_decision_rules(threshold, grade_activations, grade_domain, decision_rules, grades, grade_tuple):

    n_activated = np.sum([1 for elem in [elem.max() for elem in grade_activations] if elem > threshold])

    # Initialize
    grade_bottom = np.zeros_like(grade_domain)

    if n_activated > 0:
        # Create a figure and a set of subplots
        fig_size = 6
        fig, axes = plt.subplots(nrows=n_activated, figsize=(fig_size, fig_size * n_activated))
        if n_activated == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        row = 0

        for i, grade_activation in enumerate(grade_activations):
            if grade_activation.max() > threshold:
                # Color
                c = 'C{}'.format(grades[decision_rules[i][-1]])
                # Fill region
                axes[row].fill_between(grade_domain, grade_bottom, grade_activation, facecolor=c, alpha=0.5)
                # Fuzzy sets
                for j, a_set in enumerate(grade_tuple):
                    axes[row].plot(grade_domain, a_set, color='C{}'.format(j), linestyle='--')
                # Title
                title = 'Output for decision rule {}'.format(str(i + 1).zfill(2))
                axes[row].set_title(title)
                # Update row
                row += 1

        adjust_plot(plt, axes)

    else:

        print("There is no active decision rule for the given threshold.")


def plot_final_grade(grade_tuple, grade_domain, final_grade, final_grade_activation, c, aggregated):

    grade_bottom = np.zeros_like(grade_domain)

    # Create a figure and a subplot
    fig_size = 8
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Fuzzy sets
    for i, a_set in enumerate(grade_tuple):
        ax.plot(grade_domain, a_set, color='C{}'.format(i), linestyle='--')

    # Aggregated membership
    ax.fill_between(grade_domain, grade_bottom, aggregated, facecolor=c, alpha=0.5)

    # Final grade
    ax.plot([final_grade, final_grade], [0, final_grade_activation], 'k--')

    # Title
    rounded_grade = int(np.round(final_grade))
    title = 'Aggregated membership, Final grade: {}'.format(rounded_grade)
    ax.set_title(title)

    adjust_plot(plt, [ax])