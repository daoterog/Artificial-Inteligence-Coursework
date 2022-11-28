"""Decision rules module."""

import re
import numpy as np

DECISION_RULES = '''
[Assessment Poor] and [Homework Poor] and [Stress Low] then [Grade Worst]
[Assessment Poor] and [Homework Poor] and [Stress Medium] then [Grade Bad]
[Assessment Poor] and [Homework Poor] and [Stress High] then [Grade Bad]
[Assessment Poor] and [Homework Medium] and [Stress Low] then [Grade Good]
[Assessment Poor] and [Homework Medium] and [Stress Medium] then [Grade Good]
[Assessment Poor] and [Homework Medium] and [Stress High] then [Grade Good]
[Assessment Poor] and [Homework High] and [Stress Low] then [Grade Good]
[Assessment Poor] and [Homework High] and [Stress Medium] then [Grade Excellent]
[Assessment Poor] and [Homework High] and [Stress High] then [Grade Excellent]
[Assessment Medium] and [Homework Poor] and [Stress Low] then [Grade Bad]
[Assessment Medium] and [Homework Poor] and [Stress Medium] then [Grade Bad]
[Assessment Medium] and [Homework Poor] and [Stress High] then [Grade Mediocre]
[Assessment Medium] and [Homework Medium] and [Stress Low] then [Grade Good]
[Assessment Medium] and [Homework Medium] and [Stress Medium] then [Grade Good]
[Assessment Medium] and [Homework Medium] and [Stress High] then [Grade Good]
[Assessment Medium] and [Homework High] and [Stress Low] then [Grade Excellent]
[Assessment Medium] and [Homework High] and [Stress Medium] then [Grade Excellent]
[Assessment Medium] and [Homework High] and [Stress High] then [Grade Excellent]
[Assessment High] and [Homework Poor] and [Stress Low] then [Grade Bad]
[Assessment High] and [Homework Poor] and [Stress Medium] then [Grade Bad]
[Assessment High] and [Homework Poor] and [Stress High] then [Grade Mediocre]
[Assessment High] and [Homework Medium] and [Stress Low] then [Grade Good]
[Assessment High] and [Homework Medium] and [Stress Medium] then [Grade Good]
[Assessment High] and [Homework Medium] and [Stress High] then [Grade Excellent]
[Assessment High] and [Homework High] and [Stress Low] then [Grade Excellent]
[Assessment High] and [Homework High] and [Stress Medium] then [Grade Excellent]
[Assessment High] and [Homework High] and [Stress High] then [Grade Excellent]
'''

def get_decision_rules():

    # String
    decision_rules = DECISION_RULES.splitlines()
    decision_rules = [s for s in decision_rules if len(s) > 0]
    decision_rules = [re.findall(r'[\w]+', s) for s in decision_rules]
    decision_rules = np.vectorize(str.lower)(decision_rules)

    return decision_rules

# Operator precedence: Not, And, Or
def process_rule(rule, activation):
    # Dictionaries
    categories = {'assessment': 0, 'homework': 1, 'stress': 2}
    levels = {'low': 0, 'poor': 0, 'medium': 1, 'high': 2}
    # Operators
    operators = ['and', 'or']
    n_activations = 3
    # Activations
    activations = np.zeros((n_activations,))
    for i in range(n_activations):
        category, level, _ = rule[(i * 3):((i + 1) * 3)]
        activations[i] = activation[categories[category], levels[level]]
    # Operators
    op_1, op_2 = rule[2], rule[5]
    assert op_1 in operators and op_2 in operators
    # Active rule
    if op_1 == op_2:
        if op_1 == 'and':
            active_rule = activations.min()
        else:  # 'or'
            active_rule = activations.max()
    else:
        a, b, c = activations
        if op_1 == 'and':  # (a and b) or c
            active_rule = np.maximum(np.minimum(a, b), c)
        else:  # a or (b and c)
            active_rule = np.maximum(a, np.minimum(b, c))
    # active_rule
    return active_rule