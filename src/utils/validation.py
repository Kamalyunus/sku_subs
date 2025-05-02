#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Statistical validation for substitution analysis.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)

def validate_substitution_with_controls(sales_df, oos_df, price_df, promo_df, 
                                        item_a, item_b, control_vars=None):
    """
    Validate substitution effect with additional control variables
    
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
    logger.info(f"Validating substitution relationship between {item_a} and {item_b}")
    
    # Create analysis dataframe
    analysis_df = pd.DataFrame({
        'sales_b': sales_df[item_b],
        'oos_a': oos_df[item_a],
        'promo_a': promo_df[item_a]
    })
    
    # Add price if available
    if item_a in price_df.columns:
        analysis_df['price_a'] = price_df[item_a]
    
    if item_b in price_df.columns:
        analysis_df['price_b'] = price_df[item_b]
        
    if item_b in promo_df.columns:
        analysis_df['promo_b'] = promo_df[item_b]  # Control for item B's own promotions
    
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
        # Model 1: OOS effect
        oos_formula = 'sales_b ~ oos_a'
        
        if 'price_b' in analysis_df.columns:
            oos_formula += ' + price_b'
            
        if 'promo_b' in analysis_df.columns:
            oos_formula += ' + promo_b'
            
        # Add control variables if present
        if control_vars is not None:
            control_cols = [col for col in control_vars.columns if col in analysis_df.columns]
            if control_cols:
                oos_formula += ' + ' + ' + '.join(control_cols)
        
        model_oos = sm.formula.ols(formula=oos_formula, data=analysis_df).fit()
        
        # Model 2: Promotion effect
        promo_formula = 'sales_b ~ promo_a'
        
        if 'price_b' in analysis_df.columns:
            promo_formula += ' + price_b'
            
        if 'promo_b' in analysis_df.columns:
            promo_formula += ' + promo_b'
            
        # Add control variables if present
        if control_vars is not None:
            control_cols = [col for col in control_vars.columns if col in analysis_df.columns]
            if control_cols:
                promo_formula += ' + ' + ' + '.join(control_cols)
        
        model_promo = sm.formula.ols(formula=promo_formula, data=analysis_df).fit()
        
        # Model 3: Price effect (if price data available)
        if 'price_a' in analysis_df.columns and 'price_b' in analysis_df.columns:
            # Create log values for elasticity calculation
            analysis_df['log_sales_b'] = np.log(analysis_df['sales_b'].replace(0, 0.01))
            analysis_df['log_price_a'] = np.log(analysis_df['price_a'])
            analysis_df['log_price_b'] = np.log(analysis_df['price_b'])
            
            price_formula = 'log_sales_b ~ log_price_a + log_price_b'
            
            if 'promo_a' in analysis_df.columns:
                price_formula += ' + promo_a'
                
            if 'promo_b' in analysis_df.columns:
                price_formula += ' + promo_b'
                
            # Add control variables if present
            if control_vars is not None:
                control_cols = [col for col in control_vars.columns if col in analysis_df.columns]
                if control_cols:
                    price_formula += ' + ' + ' + '.join(control_cols)
            
            model_price = sm.formula.ols(formula=price_formula, data=analysis_df).fit()
            
            # Extract price effect (cross-price elasticity)
            price_effect = model_price.params.get('log_price_a', 0)
            price_pvalue = model_price.pvalues.get('log_price_a', 1)
            
            results['price_effect'] = price_effect
            results['price_significant'] = price_pvalue < 0.05 and price_effect > 0
            results['price_pvalue'] = price_pvalue
            results['price_model_r2'] = model_price.rsquared
        else:
            results['price_effect'] = 0
            results['price_significant'] = False
            results['price_pvalue'] = 1
            results['price_model_r2'] = 0
        
        # Get effects and p-values for OOS and promotion
        oos_effect = model_oos.params.get('oos_a', 0)
        oos_pvalue = model_oos.pvalues.get('oos_a', 1)
        
        promo_effect = model_promo.params.get('promo_a', 0)
        promo_pvalue = model_promo.pvalues.get('promo_a', 1)
        
        # Calculate relative effects
        mean_sales = analysis_df['sales_b'].mean()
        if mean_sales > 0:
            oos_relative_effect = oos_effect / mean_sales
            promo_relative_effect = -promo_effect / mean_sales  # Negative for cannibalization
        else:
            oos_relative_effect = 0
            promo_relative_effect = 0
        
        # Store results
        results.update({
            'oos_effect': oos_relative_effect,
            'oos_significant': oos_pvalue < 0.05 and oos_effect > 0,
            'oos_pvalue': oos_pvalue,
            'oos_model_r2': model_oos.rsquared,
            'promo_effect': promo_relative_effect,
            'promo_significant': promo_pvalue < 0.05 and promo_effect < 0,
            'promo_pvalue': promo_pvalue,
            'promo_model_r2': model_promo.rsquared,
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