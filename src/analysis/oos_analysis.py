#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis of out-of-stock substitution effects.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from src.utils.validation import validate_substitution_with_controls

logger = logging.getLogger(__name__)

def calculate_oos_substitution(sales_df, oos_df, price_df, items_list, min_oos_days=5, control_vars=None, promo_df=None):
    """
    Calculate substitution effect when one item is out of stock,
    controlling for price changes and time-based factors
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with dates as index and items as columns, values are sales
    oos_df : DataFrame
        Pivot table with dates as index and items as columns, values are OOS flags (0/1)
    price_df : DataFrame
        Pivot table with dates as index and items as columns, values are prices
    items_list : list
        List of item IDs to analyze
    min_oos_days : int
        Minimum number of OOS days required to consider a valid signal
    control_vars : DataFrame, optional
        Control variables for regression (e.g., weekday, month)
    promo_df : DataFrame, optional
        Pivot table with promotion flags
        
    Returns:
    --------
    tuple
        (substitution_matrix, significance_matrix)
    """
    logger.info(f"Calculating OOS substitution effects with min_oos_days={min_oos_days}")
    
    # Use float64 type for the substitution matrix to avoid dtype warnings
    substitution_matrix = pd.DataFrame(0.0, index=items_list, columns=items_list, dtype=float)
    significance_matrix = pd.DataFrame(False, index=items_list, columns=items_list)
    
    item_count = len(items_list)
    processed = 0
    
    for item_a in items_list:
        processed += 1
        if processed % 10 == 0:
            logger.info(f"OOS Analysis: Processed {processed}/{item_count} items")
        
        # Skip if item not in OOS data
        if item_a not in oos_df.columns:
            continue
            
        # Days when item A was OOS
        oos_days = oos_df[oos_df[item_a] == 1].index
        
        if len(oos_days) < min_oos_days:
            continue
            
        # Days when item A was in stock
        normal_days = oos_df[oos_df[item_a] == 0].index
        
        if len(normal_days) == 0:
            continue
        
        for item_b in items_list:
            if item_a == item_b or item_b not in sales_df.columns:
                continue
            
            if promo_df is not None:
                # Use enhanced validation with controls for more robust analysis
                validation_result = validate_substitution_with_controls(
                    sales_df, oos_df, price_df, promo_df, 
                    item_a, item_b, control_vars
                )
                
                # Extract OOS effect from validation
                if validation_result['validation_successful']:
                    oos_effect = validation_result['oos_effect']
                    oos_significant = validation_result['oos_significant']
                    
                    if oos_effect > 0 and oos_significant:
                        substitution_matrix.loc[item_a, item_b] = oos_effect
                        significance_matrix.loc[item_a, item_b] = True
                continue
            
            # If promo_df is not available, fall back to the original method
            # Create analysis dataframe
            analysis_df = pd.DataFrame({
                'sales_b': sales_df[item_b],
                'oos_a': oos_df[item_a]
            })
            
            # Add price control if available
            if item_b in price_df.columns:
                analysis_df['price_b'] = price_df[item_b]
            
            # Add control variables if provided
            if control_vars is not None:
                analysis_df = pd.concat([analysis_df, control_vars], axis=1)
            
            # Drop missing values
            analysis_df = analysis_df.dropna()
            
            if len(analysis_df) < 30:  # Not enough data
                continue
            
            try:
                # Prepare the model formula
                formula_parts = ['sales_b ~ oos_a']
                
                if 'price_b' in analysis_df.columns:
                    formula_parts.append('price_b')
                
                # Add control variables if present
                if control_vars is not None:
                    control_cols = [col for col in control_vars.columns if col in analysis_df.columns]
                    formula_parts.extend(control_cols)
                
                # Create formula
                formula = ' + '.join(formula_parts)
                
                # Run regression using formula API for more flexibility
                model = sm.formula.ols(formula=formula, data=analysis_df).fit()
                
                # Extract the OOS effect coefficient
                oos_effect = model.params.get('oos_a', 0)
                oos_pvalue = model.pvalues.get('oos_a', 1)
                
                # Calculate relative effect size
                mean_sales = analysis_df['sales_b'].mean()
                if mean_sales > 0:
                    relative_effect = oos_effect / mean_sales
                    
                    # Only count positive substitution effects that are significant
                    if relative_effect > 0 and oos_pvalue < 0.05:
                        substitution_matrix.loc[item_a, item_b] = relative_effect
                        significance_matrix.loc[item_a, item_b] = True
            except Exception as e:
                logger.warning(f"Regression failed for OOS analysis of {item_a}->{item_b}: {str(e)}")
                continue
    
    logger.info("OOS substitution analysis complete")
    return substitution_matrix, significance_matrix

def calculate_oos_substitution_with_validation(sales_df, oos_df, price_df, promo_df, items_list, min_oos_days=5, control_vars=None):
    """
    Calculate substitution effect using the enhanced validation framework
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with dates as index and items as columns, values are sales
    oos_df : DataFrame
        Pivot table with dates as index and items as columns, values are OOS flags (0/1)
    price_df : DataFrame
        Pivot table with dates as index and items as columns, values are prices
    promo_df : DataFrame
        Pivot table with promotion flags
    items_list : list
        List of item IDs to analyze
    min_oos_days : int
        Minimum number of OOS days required to consider a valid signal
    control_vars : DataFrame, optional
        Control variables for regression (e.g., weekday, month)
        
    Returns:
    --------
    tuple
        (substitution_matrix, significance_matrix, detailed_results)
    """
    logger.info(f"Calculating OOS substitution with validation (min_oos_days={min_oos_days})")
    
    substitution_matrix = pd.DataFrame(0.0, index=items_list, columns=items_list, dtype=float)
    significance_matrix = pd.DataFrame(False, index=items_list, columns=items_list)
    detailed_results = {}
    
    item_count = len(items_list)
    processed = 0
    
    for item_a in items_list:
        processed += 1
        if processed % 10 == 0:
            logger.info(f"OOS Validation: Processed {processed}/{item_count} items")
        
        # Skip if item not in OOS data
        if item_a not in oos_df.columns:
            continue
            
        # Days when item A was OOS
        oos_days = oos_df[oos_df[item_a] == 1].index
        
        if len(oos_days) < min_oos_days:
            continue
        
        detailed_results[item_a] = {}
        
        for item_b in items_list:
            if item_a == item_b or item_b not in sales_df.columns:
                continue
            
            # Use validation function from validation.py
            result = validate_substitution_with_controls(
                sales_df, oos_df, price_df, promo_df, 
                item_a, item_b, control_vars
            )
            
            detailed_results[item_a][item_b] = result
            
            if result['validation_successful'] and result['oos_significant']:
                substitution_matrix.loc[item_a, item_b] = result['oos_effect']
                significance_matrix.loc[item_a, item_b] = True
    
    logger.info("OOS substitution validation complete")
    return substitution_matrix, significance_matrix, detailed_results