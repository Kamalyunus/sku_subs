#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for extracting price elasticity from the combined model results.
This module provides utility functions to work with combined model outputs.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def extract_elasticity_from_results(combined_results):
    """
    Extract price elasticity matrices from combined model results
    
    Parameters:
    -----------
    combined_results : tuple
        Results tuple from calculate_substitution_effects
        
    Returns:
    --------
    tuple
        (elasticity_matrix, significance_matrix)
    """
    # Extract just the price elasticity matrices
    elasticity_matrix = combined_results[2]  # price_matrix is the 3rd element (index 2)
    significance_matrix = combined_results[3]  # price_significance is the 4th element (index 3)
    
    logger.info(f"Elasticity matrices extracted from combined results")
    return elasticity_matrix, significance_matrix