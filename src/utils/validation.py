#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical validation for substitution analysis.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
import math

logger = logging.getLogger(__name__)

def validate_substitution(sales_df, oos_df, price_df, promo_df, 
                          item_a, item_b, control_vars=None):
    """
    Validate substitution effect using a combined model that includes 
    OOS, price, and promo effects simultaneously
    
    Parameters:
    -----------
    sales_df : DataFrame
        Sales data in pivot format
    oos_df : DataFrame
        OOS flags in pivot format
    price_df : DataFrame
        Price data in pivot format
    promo_df : DataFrame
        Promotion flags in pivot format
    item_a : str
        Primary item ID
    item_b : str
        Potential substitute item ID
    control_vars : DataFrame
        Additional control variables (e.g., weekday, month, holidays)
        Must have the same index as sales_df
        
    Returns:
    --------
    dict
        Dictionary with validation results
    """
    logger.info(f"Validating substitution using combined model for {item_a} and {item_b}")
    
    # Create analysis dataframe
    analysis_df = pd.DataFrame({
        'sales_b': sales_df[item_b],
    })
    
    # Add OOS flags
    if item_a in oos_df.columns:
        analysis_df['oos_a'] = oos_df[item_a]
    
    if item_b in oos_df.columns:
        analysis_df['oos_b'] = oos_df[item_b]
    
    # Add price data
    if item_a in price_df.columns:
        analysis_df['price_a'] = price_df[item_a]
    
    if item_b in price_df.columns:
        analysis_df['price_b'] = price_df[item_b]
    
    # Add promotion flags
    if item_a in promo_df.columns:
        analysis_df['promo_a'] = promo_df[item_a]
    
    if item_b in promo_df.columns:
        analysis_df['promo_b'] = promo_df[item_b]
    
    # Add interaction terms between price and OOS
    if 'price_a' in analysis_df.columns and 'oos_a' in analysis_df.columns:
        analysis_df['price_a_x_oos_a'] = analysis_df['price_a'] * analysis_df['oos_a']
    
    if 'price_b' in analysis_df.columns and 'oos_b' in analysis_df.columns:
        analysis_df['price_b_x_oos_b'] = analysis_df['price_b'] * analysis_df['oos_b']
    
    # Add any additional control variables
    if control_vars is not None:
        analysis_df = pd.concat([analysis_df, control_vars], axis=1)
    
    # Drop rows with missing values
    analysis_df = analysis_df.dropna()
    
    if len(analysis_df) < 30:  # Not enough data for reliable analysis
        logger.warning(f"Not enough data for validation: {item_a}, {item_b}")
        return {
            'oos_effect': 0,
            'oos_significant': False,
            'promo_effect': 0,
            'promo_significant': False,
            'price_effect': 0,
            'price_significant': False,
            'sample_size': len(analysis_df),
            'validation_successful': False
        }
    
    results = {}
    
    try:
        # First, run a linear model to properly interpret OOS effects
        linear_formula_parts = ['sales_b ~ ']
        linear_predictor_parts = []
        
        # Add OOS effects
        if 'oos_a' in analysis_df.columns:
            linear_predictor_parts.append('oos_a')
        
        if 'oos_b' in analysis_df.columns:
            linear_predictor_parts.append('oos_b')
        
        # Add price effects
        if 'price_a' in analysis_df.columns:
            linear_predictor_parts.append('price_a')
        
        if 'price_b' in analysis_df.columns:
            linear_predictor_parts.append('price_b')
        
        # Add promo effects
        if 'promo_a' in analysis_df.columns:
            linear_predictor_parts.append('promo_a')
        
        if 'promo_b' in analysis_df.columns:
            linear_predictor_parts.append('promo_b')
        
        # Add interaction terms
        if 'price_a_x_oos_a' in analysis_df.columns:
            linear_predictor_parts.append('price_a_x_oos_a')
        
        if 'price_b_x_oos_b' in analysis_df.columns:
            linear_predictor_parts.append('price_b_x_oos_b')
        
        # Add control variables if present
        if control_vars is not None:
            control_cols = [col for col in control_vars.columns if col in analysis_df.columns]
            if control_cols:
                linear_predictor_parts.extend(control_cols)
        
        # Create linear formula
        linear_formula = linear_formula_parts[0] + ' + '.join(linear_predictor_parts)
        
        # Run the linear model for proper OOS and promo effects
        model_linear = sm.formula.ols(formula=linear_formula, data=analysis_df).fit()
        
        # Now create the log-log model for proper price elasticity
        # Create log transformations for price elasticity
        analysis_df['log_sales_b'] = np.log(analysis_df['sales_b'].replace(0, 0.01))
        
        if 'price_a' in analysis_df.columns:
            analysis_df['log_price_a'] = np.log(analysis_df['price_a'])
        
        if 'price_b' in analysis_df.columns:
            analysis_df['log_price_b'] = np.log(analysis_df['price_b'])
        
        # Build log-log model
        log_formula_parts = ['log_sales_b ~ ']
        log_predictor_parts = []
        
        # Use logged prices
        if 'log_price_a' in analysis_df.columns:
            log_predictor_parts.append('log_price_a')
        
        if 'log_price_b' in analysis_df.columns:
            log_predictor_parts.append('log_price_b')
        
        # Add OOS effects
        if 'oos_a' in analysis_df.columns:
            log_predictor_parts.append('oos_a')
        
        if 'oos_b' in analysis_df.columns:
            log_predictor_parts.append('oos_b')
        
        # Add promo effects
        if 'promo_a' in analysis_df.columns:
            log_predictor_parts.append('promo_a')
        
        if 'promo_b' in analysis_df.columns:
            log_predictor_parts.append('promo_b')
        
        # Add interaction terms
        if 'price_a_x_oos_a' in analysis_df.columns:
            log_predictor_parts.append('price_a_x_oos_a')
        
        if 'price_b_x_oos_b' in analysis_df.columns:
            log_predictor_parts.append('price_b_x_oos_b')
        
        # Add control variables if present
        if control_vars is not None:
            control_cols = [col for col in control_vars.columns if col in analysis_df.columns]
            if control_cols:
                log_predictor_parts.extend(control_cols)
        
        # Create log-log formula
        log_formula = log_formula_parts[0] + ' + '.join(log_predictor_parts)
        
        # Run the log-log model for proper price elasticity
        model_log = sm.formula.ols(formula=log_formula, data=analysis_df).fit()
        
        # Extract effects and p-values
        # OOS and promo effects from linear model
        oos_effect = model_linear.params.get('oos_a', 0) if 'oos_a' in model_linear.params else 0
        oos_pvalue = model_linear.pvalues.get('oos_a', 1) if 'oos_a' in model_linear.pvalues else 1
        
        promo_effect = model_linear.params.get('promo_a', 0) if 'promo_a' in model_linear.params else 0
        promo_pvalue = model_linear.pvalues.get('promo_a', 1) if 'promo_a' in model_linear.pvalues else 1
        
        # Price elasticity from log-log model 
        price_effect = model_log.params.get('log_price_a', 0) if 'log_price_a' in model_log.params else 0
        price_pvalue = model_log.pvalues.get('log_price_a', 1) if 'log_price_a' in model_log.pvalues else 1
        
        # For OOS in log-log model, convert semi-elasticity to percentage effect
        # The coefficient in a log-linear model gives the approximate percentage change
        # For binary variables, the exact percentage change is (e^coefficient - 1) * 100
        if 'oos_a' in model_log.params:
            oos_log_coef = model_log.params.get('oos_a', 0)
            oos_log_effect = (math.exp(oos_log_coef) - 1) 
            oos_log_pvalue = model_log.pvalues.get('oos_a', 1)
            # Store log model OOS effect for reference
            results['oos_log_effect'] = oos_log_effect
            results['oos_log_pvalue'] = oos_log_pvalue
        
        # For promo in log-log model
        if 'promo_a' in model_log.params:
            promo_log_coef = model_log.params.get('promo_a', 0)
            promo_log_effect = (math.exp(promo_log_coef) - 1)
            promo_log_pvalue = model_log.pvalues.get('promo_a', 1)
            # Store log model promo effect for reference
            results['promo_log_effect'] = promo_log_effect
            results['promo_log_pvalue'] = promo_log_pvalue
        
        # Calculate relative effects for the linear model
        mean_sales = analysis_df['sales_b'].mean()
        if mean_sales > 0:
            oos_relative_effect = oos_effect / mean_sales  
            promo_relative_effect = -promo_effect / mean_sales  # Negative for cannibalization
        else:
            oos_relative_effect = 0
            promo_relative_effect = 0
        
        # Store the results
        results.update({
            'oos_effect': oos_relative_effect,
            'oos_significant': oos_pvalue < 0.05 and oos_effect > 0,
            'oos_pvalue': oos_pvalue,
            'oos_coef': oos_effect,
            'promo_effect': promo_relative_effect,
            'promo_significant': promo_pvalue < 0.05 and promo_effect < 0,
            'promo_pvalue': promo_pvalue,
            'promo_coef': promo_effect,
            'price_effect': price_effect,  # This is elasticity from log-log model
            'price_significant': price_pvalue < 0.05 and price_effect > 0,
            'price_pvalue': price_pvalue,
            'linear_model_r2': model_linear.rsquared,
            'log_model_r2': model_log.rsquared,
            'sample_size': len(analysis_df),
            'validation_successful': True
        })
        
    except Exception as e:
        logger.error(f"Validation failed for {item_a}->{item_b}: {str(e)}")
        results = {
            'oos_effect': 0,
            'oos_significant': False,
            'promo_effect': 0,
            'promo_significant': False,
            'price_effect': 0,
            'price_significant': False,
            'sample_size': len(analysis_df),
            'validation_successful': False,
            'error': str(e)
        }
    
    return results