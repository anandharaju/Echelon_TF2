from config import constants as cnst


class SectionActivationDistribution:
    a_activation_magnitude = {cnst.LEAK: 0}  # All Samples Section Magnitude
    b_activation_magnitude = {cnst.LEAK: 0}  # Benign Samples Section Magnitude
    m_activation_magnitude = {cnst.LEAK: 0}  # Malware Samples Section Magnitude

    a_activation_histogram = {cnst.LEAK: 0}  # All Samples Section Distribution
    b_activation_histogram = {cnst.LEAK: 0}  # Benign Samples Section Distribution
    m_activation_histogram = {cnst.LEAK: 0}  # Malware Samples Section Distribution

    a_section_support = {}       # Presence of each section among all samples
    b_section_support = {}       # Presence of each section among all benign samples
    m_section_support = {}       # Presence of each section among all malware samples

    b1_count = None
    b1_b_truth_count = None
    b1_m_truth_count = None