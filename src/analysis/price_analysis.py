#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for interpreting results from the combined model.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

def add_interpretations(result):
    """
    Add human-readable interpretations to combined model results
    
    Parameters:
    -----------
    result : dict
        Result dictionary from validate_substitution or detailed_results
        
    Returns:
    --------
    dict
        The same result dictionary with added interpretations
    """
    if not result.get('validation_successful', False):
        return result
    
    # Price elasticity interpretation
    price_effect = result.get('price_effect', 0)
    if price_effect > 0 and result.get('price_significant', False):
        result['price_interpretation'] = "Substitute relationship (significant)"
    elif price_effect > 0 and not result.get('price_significant', False):
        result['price_interpretation'] = "Potential substitute (not significant)"
    elif price_effect < 0 and result.get('price_significant', False):
        result['price_interpretation'] = "Complement relationship (significant)"
    elif price_effect < 0 and not result.get('price_significant', False):
        result['price_interpretation'] = "Potential complement (not significant)"
    else:
        result['price_interpretation'] = "No price relationship detected"
    
    # OOS effect interpretation
    oos_effect = result.get('oos_effect', 0)
    if oos_effect > 0 and result.get('oos_significant', False):
        result['oos_interpretation'] = "Strong substitution (significant)"
    elif oos_effect > 0 and not result.get('oos_significant', False):
        result['oos_interpretation'] = "Weak substitution (not significant)"
    elif oos_effect < 0 and result.get('oos_significant', False):
        result['oos_interpretation'] = "Cannibalization (significant)"
    elif oos_effect < 0 and not result.get('oos_significant', False):
        result['oos_interpretation'] = "Potential cannibalization (not significant)"
    else:
        result['oos_interpretation'] = "No OOS relationship detected"
    
    # Promo effect interpretation
    promo_effect = result.get('promo_effect', 0)
    if promo_effect < 0 and result.get('promo_significant', False):  # Negative for cannibalization
        result['promo_interpretation'] = "Strong cannibalization (significant)"
    elif promo_effect < 0 and not result.get('promo_significant', False):
        result['promo_interpretation'] = "Weak cannibalization (not significant)"
    elif promo_effect > 0 and result.get('promo_significant', False):
        result['promo_interpretation'] = "Halo effect (significant)"
    elif promo_effect > 0 and not result.get('promo_significant', False):
        result['promo_interpretation'] = "Potential halo effect (not significant)"
    else:
        result['promo_interpretation'] = "No promotion relationship detected"
    
    # Overall relationship categorization
    if (result.get('price_significant', False) and price_effect > 0) or (result.get('oos_significant', False) and oos_effect > 0):
        result['relationship_type'] = "Substitute"
    elif (result.get('price_significant', False) and price_effect < 0) or (result.get('promo_significant', False) and promo_effect < 0):
        result['relationship_type'] = "Complement"
    else:
        result['relationship_type'] = "Undefined"
    
    return result