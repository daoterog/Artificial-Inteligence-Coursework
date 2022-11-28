"""Membership functions module."""

import numpy as np
import skfuzzy as fuzz

three_sd = 3

def gauss_parameters(mean, zero_value):
    sigma = np.abs(zero_value - mean) / three_sd
    return mean, sigma

def sigmoid_parameters(x, a, b):
    c = -np.log((1 - a) / a) / (x - b)
    return b, c

def get_codemf(domain: np.ndarray):

    # Poor
    poor_code_quality = fuzz.trapmf(x=domain, abcd=[0, 10, 20, 30])

    # Medium
    mean, sigma = gauss_parameters(mean=50, zero_value=10)
    avg_code_quality = fuzz.gaussmf(x=domain, mean=mean, sigma=sigma)

    # High
    b, c = sigmoid_parameters(x=70, a=0.05, b=75)
    high_code_quality = fuzz.sigmf(x=domain, b=b, c=c)

    return poor_code_quality, avg_code_quality, high_code_quality

def get_reportmf(domain: np.ndarray):

    # Poor
    b, c = sigmoid_parameters(x=30, a=0.005, b=25)
    poor_report_quality = fuzz.sigmf(x=domain, b=b, c=c)

    # Medium
    mean, sigma = gauss_parameters(mean=35, zero_value=10)
    avg_report_quality = fuzz.gaussmf(x=domain, mean=mean, sigma=sigma)

    # High
    high_report_quality = fuzz.trapmf(x=domain, abcd=[40, 45, 50, 50])

    return poor_report_quality, avg_report_quality, high_report_quality

def get_punctualitymf(domain: np.ndarray):

    # Low
    b, c = sigmoid_parameters(x=5, a=0.005, b=3)
    # stress_lo = fuzz.sigmf(x=punctuality_domain, b=b, c=c)
    poor_punctuality = fuzz.trapmf(x=domain, abcd=[2, 4, 5, 7])

    # Medium
    mean, sigma = gauss_parameters(mean=0.5, zero_value=2)
    avg_punctuality = fuzz.gaussmf(x=domain, mean=mean, sigma=sigma)

    # High
    b, c = sigmoid_parameters(x=5, a=0.005, b=7)
    # stress_hi = fuzz.sigmf(x=punctuality_domain, b=b, c=c)
    high_punctuality = fuzz.trapmf(x=domain, abcd=[-3, -2, -1, 0])

    return poor_punctuality, avg_punctuality, high_punctuality

def get_grademf(domain: np.ndarray):

    # Worst
    mean, sigma = gauss_parameters(mean=10, zero_value=20)
    grade_wo = fuzz.gaussmf(x=domain, mean=mean, sigma=sigma)

    # Bad
    mean, sigma = gauss_parameters(mean=19, zero_value=27)
    grade_bd = fuzz.gaussmf(x=domain, mean=mean, sigma=sigma)

    # Medium
    mean, sigma = gauss_parameters(mean=30, zero_value=25)
    grade_md = fuzz.gaussmf(x=domain, mean=mean, sigma=sigma)

    # Good
    mean, sigma = gauss_parameters(mean=37, zero_value=33)
    grade_gd = fuzz.gaussmf(x=domain, mean=mean, sigma=sigma)

    # Excellent
    mean, sigma = gauss_parameters(mean=45, zero_value=49)
    grade_ex = fuzz.gaussmf(x=domain, mean=mean, sigma=sigma)

    return grade_wo, grade_bd, grade_md, grade_gd, grade_ex