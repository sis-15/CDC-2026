import numpy as np
from scipy.spatial.distance import jensenshannon

# Made with Gemini

def calculate_spatial_entropy(probabilities):
    """compute shannon entropy to measure inequality chaos"""
    p = np.array(probabilities)
    p = p[p > 0]  # filter zeros
    return -np.sum(p * np.log2(p))

def calculate_jsd(local_dist, national_dist):
    """compute jensen-shannon divergence between local and national complaint patterns"""
    p = np.array(local_dist, dtype=float)
    q = np.array(national_dist, dtype=float)
    
    # normalize to sum to 1
    p /= p.sum()
    q /= q.sum()
    
    return float(jensenshannon(p, q) ** 2)  # return divergence (squared distance)