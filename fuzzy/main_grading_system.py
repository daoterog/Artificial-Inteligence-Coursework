"""Main module for the gradding system"""

import numpy as np
import skfuzzy as fuzz

from decision_rules import get_decision_rules, process_rule
from membership_functions import get_codemf, get_reportmf, get_punctualitymf, get_grademf
from plotting import plot_membership_functions, plot_active_decision_rules, plot_final_grade


def fuzzy_grading_system(code_quality, punctuality, report_quality):

    # Define input domains
    code_quality_domain = np.linspace(start=0, stop=100, num=500)
    report_quality_domain = np.linspace(start=0, stop=50, num=500)
    punctuality_domain = np.linspace(start=-3, stop=7, num=500)

    # Define output domain
    grade_domain = np.linspace(start=0, stop=50, num=500)

    # Get membership functions
    poor_code_quality, avg_code_quality, high_code_quality = get_codemf(code_quality_domain)
    poor_report_quality, avg_report_quality, high_report_quality = get_reportmf(report_quality_domain)
    poor_punctuality, avg_punctuality, high_punctuality = get_punctualitymf(punctuality_domain)
    grade_wo, grade_bd, grade_md, grade_gd, grade_ex = get_grademf(grade_domain)

    code_tuple = (code_quality_domain, poor_code_quality, avg_code_quality, high_code_quality)
    report_tuple = (report_quality_domain, poor_report_quality, avg_report_quality, high_report_quality)
    punctuality_tuple = (punctuality_domain, poor_punctuality, avg_punctuality, high_punctuality)
    grade_tuple = (grade_domain, grade_wo, grade_bd, grade_md, grade_gd, grade_ex)

    # Visualize membership functions
    plot_membership_functions(code_tuple, report_tuple, punctuality_tuple, grade_tuple)

    # Define Arguments
    domain = [code_quality_domain, report_quality_domain, punctuality_domain]

    membership_functions = np.array([
        [poor_code_quality, avg_code_quality, high_code_quality],
        [poor_report_quality, avg_report_quality, high_report_quality],
        [poor_punctuality, avg_punctuality, high_punctuality],
    ])

    inputs = [code_quality, report_quality, punctuality]

    # Activation
    n = len(domain)
    activation = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            activation[i, j] = fuzz.interp_membership(x=domain[i], xmf=membership_functions[i, j], xx=inputs[i])

    print(np.round(activation, 2))

    # Get Decision Rules
    decision_rules = get_decision_rules()

    # Grades
    grades = {'worst': 0, 'bad': 1, 'mediocre': 2, 'good': 3, 'excellent': 4}

    # Get active rules
    active_rules = [process_rule(rule, activation) for rule in decision_rules]
    active_rules = np.array(active_rules)

    # Grade activations
    grade_activations = []

    # Grades
    grade_sets = [grade_wo, grade_bd, grade_md, grade_gd, grade_ex]

    for i, rule in enumerate(decision_rules):
        # Get consequent
        consequent = grade_sets[grades[rule[-1]]]
        # Grade activation
        grade_activations += [np.fmin(active_rules[i], consequent)]

    # How many decisions rules are activated?
    threshold = 0.02

    plot_active_decision_rules(threshold, grade_activations, grade_domain, decision_rules, grades, grade_sets)

    # Aggregate all output membership functions
    aggregated = np.zeros(grade_activations[0].shape)

    # Get color for the rule with the largest activation
    max_index = None
    max_value = 0

    for i, grade_activation in enumerate(grade_activations):
        grade_max = grade_activation.max()
        # Update maximum value
        if max_value < grade_max:
            max_index = i
            max_value = grade_max
        aggregated = np.fmax(aggregated, grade_activation)

    # Defuzzify result
    final_grade = fuzz.defuzz(grade_domain, aggregated, 'lom')
    print(final_grade)
    final_grade_activation = fuzz.interp_membership(grade_domain, aggregated, final_grade)

    # Plot final grade activations
    c = 'C{}'.format(grades[decision_rules[max_index][-1]])
    plot_final_grade(grade_sets, grade_domain, final_grade, final_grade_activation, c, aggregated)


if __name__ == '__main__':
    # Define inputs
    code_quality = 90
    punctuality = 2
    report_quality = 40

    # Run fuzzy grading system
    fuzzy_grading_system(code_quality, punctuality, report_quality)