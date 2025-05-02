#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analysis of price effects, including promotions, price matching, and cross-price elasticity.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from src.analysis.elasticity import calculate_cross_price_elasticity, calculate_elasticity_matrix

logger = logging.getLogger(__name__)

def calculate_price_effects(sales_df, price_df, promo_df, price_change_types, items_list, 
                            min_price_changes=5, control_vars=None, use_elasticity=True, oos_df=None):
    """
    Calculate price effects including both promotional and competitive price matching
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with dates as index and items as columns, values are sales
    price_df : DataFrame
        Pivot table with dates as index and items as columns, values are prices
    promo_df : DataFrame
        Pivot table with dates as index and items as columns, values are promo flags (0/1)
    price_change_types : DataFrame
        Price change categorization from separate_price_changes
    items_list : list
        List of item IDs to analyze
    min_price_changes : int
        Minimum number of price change periods required for analysis
    control_vars : DataFrame, optional
        Control variables for regression (e.g., weekday, month)
    use_elasticity : bool, optional
        Whether to use elasticity-based calculation (default: True)
    oos_df : DataFrame, optional
        Pivot table with dates as index and items as columns, values are OOS flags (0/1)
        Used as a control variable to isolate price effects from availability effects
        
    Returns:
    --------
    tuple
        (price_effect_matrix, price_effect_type, significance_matrix)
    """
    logger.info(f"Calculating price effects with min_price_changes={min_price_changes}")
    
    # Check if we should use elasticity-based calculation
    if use_elasticity:
        logger.info("Using elasticity-based price effect calculation")
        return calculate_price_effects_with_elasticity(
            sales_df, price_df, promo_df, items_list, control_vars, oos_df
        )
    
    # Use float64 type for the price effect matrix to avoid dtype warnings
    price_effect_matrix = pd.DataFrame(0.0, index=items_list, columns=items_list, dtype=float) 
    price_effect_type = pd.DataFrame("none", index=items_list, columns=items_list)
    significance_matrix = pd.DataFrame(False, index=items_list, columns=items_list)
    
    item_count = len(items_list)
    processed = 0
    
    for item_a in items_list:
        processed += 1
        if processed % 10 == 0:
            logger.info(f"Price Analysis: Processed {processed}/{item_count} items")
            
        # Skip if item not in price data or price change types
        if item_a not in price_df.columns or item_a not in price_change_types.columns:
            continue
        
        # Get price changes by type
        price_changes = price_change_types[item_a]
        
        # Get days with promotion price changes
        promo_days = price_changes[price_changes == "promotion"].index
        
        # Get days with price matching changes
        matching_days = price_changes[price_changes == "price_matching"].index
        
        # Skip if not enough data
        if len(promo_days) + len(matching_days) < min_price_changes:
            continue
        
        # Get normal pricing days
        normal_days = price_changes[price_changes == "normal"].index
        
        for item_b in items_list:
            if item_a == item_b or item_b not in sales_df.columns:
                continue
            
            # Create analysis dataframe
            analysis_df = pd.DataFrame({
                'sales_b': sales_df[item_b],
                'price_a': price_df[item_a],
                'is_promo_a': (price_changes == "promotion").astype(int),
                'is_matching_a': (price_changes == "price_matching").astype(int)
            })
            
            # Add item B's price and promotion status as controls
            if item_b in price_df.columns:
                analysis_df['price_b'] = price_df[item_b]
                
            if item_b in promo_df.columns:
                analysis_df['is_promo_b'] = promo_df[item_b]
                
            # Add OOS status for both items as controls if available
            if oos_df is not None:
                if item_a in oos_df.columns:
                    analysis_df['is_oos_a'] = oos_df[item_a]
                    # Add interaction term between price and OOS
                    if 'price_a' in analysis_df.columns:
                        analysis_df['price_a_x_oos_a'] = analysis_df['price_a'] * analysis_df['is_oos_a']
                
                if item_b in oos_df.columns:
                    analysis_df['is_oos_b'] = oos_df[item_b]
                    # Add interaction term between price and OOS
                    if 'price_b' in analysis_df.columns:
                        analysis_df['price_b_x_oos_b'] = analysis_df['price_b'] * analysis_df['is_oos_b']
                
            # Add control variables if provided
            if control_vars is not None:
                analysis_df = pd.concat([analysis_df, control_vars], axis=1)
            
            # Drop missing values
            analysis_df = analysis_df.dropna()
            
            if len(analysis_df) < 30:  # Not enough data
                continue
            
            try:
                # Prepare formula components
                base_controls = []
                if 'price_b' in analysis_df.columns:
                    base_controls.append('price_b')
                if 'is_promo_b' in analysis_df.columns:
                    base_controls.append('is_promo_b')
                
                # Add OOS controls if present
                if 'is_oos_a' in analysis_df.columns:
                    base_controls.append('is_oos_a')
                    # Add interaction term
                    if 'price_a_x_oos_a' in analysis_df.columns:
                        base_controls.append('price_a_x_oos_a')
                
                if 'is_oos_b' in analysis_df.columns:
                    base_controls.append('is_oos_b')
                    # Add interaction term
                    if 'price_b_x_oos_b' in analysis_df.columns:
                        base_controls.append('price_b_x_oos_b')
                
                # Add time-based controls if present
                if control_vars is not None:
                    control_cols = [col for col in control_vars.columns if col in analysis_df.columns]
                    base_controls.extend(control_cols)
                
                # Model 1: Effect of promotions
                promo_formula = 'sales_b ~ is_promo_a'
                if base_controls:
                    promo_formula += ' + ' + ' + '.join(base_controls)
                
                promo_model = sm.formula.ols(formula=promo_formula, data=analysis_df).fit()
                
                # Model 2: Effect of price matching
                match_formula = 'sales_b ~ is_matching_a'
                if base_controls:
                    match_formula += ' + ' + ' + '.join(base_controls)
                
                match_model = sm.formula.ols(formula=match_formula, data=analysis_df).fit()
                
                # Extract coefficients
                promo_effect = promo_model.params.get('is_promo_a', 0)
                promo_pvalue = promo_model.pvalues.get('is_promo_a', 1)
                
                match_effect = match_model.params.get('is_matching_a', 0)
                match_pvalue = match_model.pvalues.get('is_matching_a', 1)
                
                # Calculate relative effects
                mean_sales = analysis_df['sales_b'].mean()
                if mean_sales > 0:
                    promo_relative = promo_effect / mean_sales
                    match_relative = match_effect / mean_sales
                    
                    # Check significance and direction
                    promo_significant = promo_pvalue < 0.05 and promo_relative < 0
                    match_significant = match_pvalue < 0.05 and match_relative < 0
                    
                    if promo_significant and match_significant:
                        # Choose stronger effect
                        if abs(promo_relative) > abs(match_relative):
                            price_effect_matrix.loc[item_a, item_b] = -promo_relative  # Negative to positive for substitution
                            price_effect_type.loc[item_a, item_b] = "promotion"
                        else:
                            price_effect_matrix.loc[item_a, item_b] = -match_relative
                            price_effect_type.loc[item_a, item_b] = "price_matching"
                        significance_matrix.loc[item_a, item_b] = True
                    elif promo_significant:
                        price_effect_matrix.loc[item_a, item_b] = -promo_relative
                        price_effect_type.loc[item_a, item_b] = "promotion"
                        significance_matrix.loc[item_a, item_b] = True
                    elif match_significant:
                        price_effect_matrix.loc[item_a, item_b] = -match_relative
                        price_effect_type.loc[item_a, item_b] = "price_matching"
                        significance_matrix.loc[item_a, item_b] = True
            
            except Exception as e:
                logger.warning(f"Regression failed for price analysis of {item_a}->{item_b}: {str(e)}")
                continue
    
    logger.info("Price effect analysis complete")
    return price_effect_matrix, price_effect_type, significance_matrix

def calculate_price_effects_with_elasticity(sales_df, price_df, promo_df, items_list, control_vars=None, oos_df=None):
    """
    Calculate price effects using cross-price elasticity from elasticity.py
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with dates as index and items as columns, values are sales
    price_df : DataFrame
        Pivot table with dates as index and items as columns, values are prices
    promo_df : DataFrame
        Pivot table with dates as index and items as columns, values are promo flags (0/1)
    items_list : list
        List of item IDs to analyze
    control_vars : DataFrame, optional
        Control variables for regression (e.g., weekday, month)
    oos_df : DataFrame, optional
        Pivot table with dates as index and items as columns, values are OOS flags (0/1)
        Used to control for and filter out OOS periods in elasticity calculations
        
    Returns:
    --------
    tuple
        (elasticity_matrix, elasticity_type, significance_matrix)
    """
    logger.info("Calculating cross-price elasticity between items")
    
    # Calculate elasticity matrix using elasticity.py function
    elasticity_matrix, significance_matrix = calculate_elasticity_matrix(
        sales_df, price_df, promo_df, items_list, control_vars, oos_df
    )
    
    # Create a matrix for elasticity type (substitutes vs complements)
    elasticity_type = pd.DataFrame("none", index=items_list, columns=items_list)
    
    # Positive elasticity = substitutes, negative = complements
    for item_a in items_list:
        for item_b in items_list:
            if item_a == item_b:
                continue
                
            if elasticity_matrix.loc[item_a, item_b] > 0:
                elasticity_type.loc[item_a, item_b] = "substitute"
            elif elasticity_matrix.loc[item_a, item_b] < 0:
                elasticity_type.loc[item_a, item_b] = "complement"
    
    logger.info("Elasticity-based price effect analysis complete")
    return elasticity_matrix, elasticity_type, significance_matrix

def analyze_individual_elasticity(sales_df, price_df, promo_df, item_a, item_b, control_vars=None, oos_df=None):
    """
    Perform detailed elasticity analysis for a specific item pair
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with sales data
    price_df : DataFrame
        Pivot table with price data
    promo_df : DataFrame
        Pivot table with promotion flags
    item_a : str
        Primary item
    item_b : str
        Potential substitute item
    control_vars : DataFrame, optional
        Control variables
    oos_df : DataFrame, optional
        Pivot table with OOS flags to control for availability
        
    Returns:
    --------
    dict
        Detailed elasticity results
    """
    logger.info(f"Analyzing elasticity relationship between {item_a} and {item_b}")
    
    result = calculate_cross_price_elasticity(
        sales_df, price_df, promo_df, item_a, item_b, control_vars, oos_df
    )
    
    # Add interpretation
    if result['status'] == 'success':
        elasticity = result['elasticity']
        if elasticity > 0 and result['significant']:
            result['interpretation'] = "Substitute relationship (significant)"
        elif elasticity > 0 and not result['significant']:
            result['interpretation'] = "Potential substitute (not significant)"
        elif elasticity < 0 and result['significant']:
            result['interpretation'] = "Complement relationship (significant)"
        elif elasticity < 0 and not result['significant']:
            result['interpretation'] = "Potential complement (not significant)"
        else:
            result['interpretation'] = "No relationship detected"
    
    return result