#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Consolidated substitution analysis module.
This streamlined module contains the essential functionality for identifying product substitutes.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import logging
import yaml
from datetime import datetime

logger = logging.getLogger(__name__)

# ---- Data Loading and Processing ----

def load_config(config_path):
    """
    Load configuration from YAML file
    
    Parameters:
    -----------
    config_path : str
        Path to config file
        
    Returns:
    --------
    dict
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as config_file:
            config = yaml.safe_load(config_file)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise

def ensure_dir(directory):
    """
    Ensure directory exists, create if not
    
    Parameters:
    -----------
    directory : str
        Directory path
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def load_transaction_data(filepath):
    """
    Load transaction data from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to transaction data CSV
        
    Returns:
    --------
    DataFrame
        Transaction data with expected columns
    """
    logger.info(f"Loading transaction data from {filepath}")
    
    # Define required columns
    minimal_required_columns = [
        'date', 'item_id', 'sales', 'price', 
        'is_on_promotion', 'is_out_of_stock'
    ]
    
    try:
        df = pd.read_csv(filepath, parse_dates=['date'])
        
        # Check for minimum required columns
        missing_cols = set(minimal_required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Input file missing required columns: {missing_cols}")
                
        logger.info(f"Loaded {len(df)} transaction records")
        return df
    
    except Exception as e:
        logger.error(f"Error loading transaction data: {str(e)}")
        raise

def load_product_attributes(filepath):
    """
    Load product attributes from CSV file
    
    Parameters:
    -----------
    filepath : str
        Path to product attributes CSV
        
    Returns:
    --------
    DataFrame
        Product attributes with item_id as key
    """
    logger.info(f"Loading product attributes from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        
        # Check for required columns
        if 'item_id' not in df.columns:
            logger.error("Product attributes file must contain 'item_id' column")
            raise ValueError("Product attributes file must contain 'item_id' column")
        
        logger.info(f"Loaded attributes for {len(df)} products")
        return df
    
    except Exception as e:
        logger.error(f"Error loading product attributes: {str(e)}")
        raise

def validate_and_preprocess(df):
    """
    Validate, preprocess, and detect anomalies in transaction data
    
    Parameters:
    -----------
    df : DataFrame
        Raw transaction data
        
    Returns:
    --------
    tuple
        (Preprocessed DataFrame, Dictionary with anomaly reports)
    """
    logger.info("Starting data validation and preprocessing")
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Initialize anomalies dictionary
    anomalies = {}
    
    # Ensure date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        logger.info("Converting date column to datetime")
        df['date'] = pd.to_datetime(df['date'])
    
    # Add essential time-based features
    logger.info("Adding time-based features")
    df['weekday'] = df['date'].dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)
    
    # Handle missing values
    df['is_on_promotion'] = df['is_on_promotion'].fillna(0).astype(int)
    df['is_out_of_stock'] = df['is_out_of_stock'].fillna(0).astype(int)
    
    # Detect and handle negative prices
    neg_prices = df[df['price'] <= 0]
    if len(neg_prices) > 0:
        logger.warning(f"Found {len(neg_prices)} records with invalid prices (<=0)")
        anomalies['negative_prices'] = {
            'count': len(neg_prices),
            'items': neg_prices['item_id'].unique().tolist()
        }
        # Remove negative prices
        df = df[df['price'] > 0]
    
    # Detect and handle negative sales
    neg_sales = df[df['sales'] < 0]
    if len(neg_sales) > 0:
        logger.warning(f"Found {len(neg_sales)} records with negative sales")
        anomalies['negative_sales'] = {
            'count': len(neg_sales),
            'items': neg_sales['item_id'].unique().tolist()
        }
        # Remove negative sales
        df = df[df['sales'] >= 0]
    
    # Drop rows where sales is NaN
    df = df.dropna(subset=['sales'])
    
    logger.info(f"Preprocessing complete. Found {len(anomalies)} types of anomalies.")
    logger.info(f"Returning {len(df)} records")
    return df, anomalies

def filter_sparse_items(transactions_df, min_days=30):
    """
    Filter out items with sparse data
    
    Parameters:
    -----------
    transactions_df : DataFrame
        Transaction data
    min_days : int
        Minimum number of days with data
        
    Returns:
    --------
    DataFrame
        Filtered transaction data
    """
    # Count days per item
    item_counts = transactions_df.groupby('item_id')['date'].nunique().rename('days_count')
    
    # Filter items with sufficient data
    valid_items = item_counts[item_counts >= min_days].index.tolist()
    
    # Filter transactions
    filtered_df = transactions_df[transactions_df['item_id'].isin(valid_items)]
    
    logger.info(f"Filtered from {transactions_df['item_id'].nunique()} to {len(valid_items)} items")
    logger.info(f"Retained {len(filtered_df)/len(transactions_df):.1%} of transaction records")
    
    return filtered_df

def create_feature_set(transactions_df, items_list, baseline_window=30, min_periods=7):
    """
    Create feature pivots for substitution analysis
    
    Parameters:
    -----------
    transactions_df : DataFrame
        Preprocessed transaction data
    items_list : list
        List of items to analyze
    baseline_window : int
        Window size for baseline price calculation
    min_periods : int
        Minimum periods required for baseline calculation
        
    Returns:
    --------
    dict
        Dictionary with pivots for sales, price, promo, oos, and controls
    """
    logger.info("Creating feature pivots")
    results = {}
    
    # Create sales pivot
    sales_pivot = transactions_df.pivot_table(
        index='date', 
        columns='item_id', 
        values='sales',
        fill_value=0
    )
    results['sales_pivot'] = sales_pivot
    
    # Create price pivot
    price_pivot = transactions_df.pivot_table(
        index='date', 
        columns='item_id', 
        values='price'
    )
    results['price_pivot'] = price_pivot
    
    # Create promotion pivot
    promo_pivot = transactions_df.pivot_table(
        index='date', 
        columns='item_id', 
        values='is_on_promotion',
        fill_value=0
    )
    results['promo_pivot'] = promo_pivot
    
    # Create OOS pivot
    oos_pivot = transactions_df.pivot_table(
        index='date', 
        columns='item_id', 
        values='is_out_of_stock',
        fill_value=0
    )
    results['oos_pivot'] = oos_pivot
    
    # Create simplified control variables pivot
    control_pivot = pd.DataFrame(index=sales_pivot.index)
    
    # Add is_weekend
    control_pivot['is_weekend'] = transactions_df.groupby('date')['is_weekend'].first()
    
    # Add weekday dummies
    for day in range(7):
        control_pivot[f'weekday_{day}'] = (transactions_df.groupby('date')['weekday'].first() == day).astype(int)
    
    results['control_pivot'] = control_pivot
    
    logger.info("Feature pivots created successfully")
    return results

def create_control_variables(sales_pivot, control_pivot):
    """
    Create enhanced control variables for regression analyses
    
    Parameters:
    -----------
    sales_pivot : DataFrame
        Sales pivot table with date index
    control_pivot : DataFrame
        Basic control variables
        
    Returns:
    --------
    DataFrame
        Enhanced control variables
    """
    logger.info("Creating control variables")
    
    # Extract unique dates from the pivot
    unique_dates = sales_pivot.index.get_level_values('date').unique()
    
    # Add quarter indicators if not present
    enhanced_controls = control_pivot.copy()
    if not any(col.startswith('quarter_') for col in enhanced_controls.columns):
        quarter_dummies = pd.get_dummies(pd.DatetimeIndex(unique_dates).quarter, prefix='quarter')
        quarter_dummies.index = unique_dates
        for col in quarter_dummies.columns:
            enhanced_controls[col] = quarter_dummies[col].reindex(enhanced_controls.index).values
    
    return enhanced_controls

# ---- Substitution Analysis ----

def check_substitution_scope(item_a, item_b, product_attributes, substitution_scope="category"):
    """
    Check if two items are within the same substitution scope
    
    Parameters:
    -----------
    item_a : str
        First item ID
    item_b : str
        Second item ID
    product_attributes : DataFrame
        DataFrame with product attribute data
    substitution_scope : str
        Scope for substitution filtering: "category", "sub_category", or "all"
        
    Returns:
    --------
    bool
        True if items are in same category/subcategory (based on scope)
    """
    # If no filtering or no attributes data, all pairs are valid
    if substitution_scope == "all" or product_attributes is None:
        return True
    
    # Make sure both items exist in attributes
    item_a_exists = item_a in product_attributes['item_id'].values
    item_b_exists = item_b in product_attributes['item_id'].values
    
    if not (item_a_exists and item_b_exists):
        return False
    
    # Get attributes for each item
    item_a_attr = product_attributes[product_attributes['item_id'] == item_a].iloc[0]
    item_b_attr = product_attributes[product_attributes['item_id'] == item_b].iloc[0]
    
    # Check if they're in the same category/subcategory
    if substitution_scope == "category":
        if 'category' not in item_a_attr or 'category' not in item_b_attr:
            return False
        return item_a_attr['category'] == item_b_attr['category']
    
    elif substitution_scope == "sub_category":
        if 'sub_category' not in item_a_attr or 'sub_category' not in item_b_attr:
            return False
        return item_a_attr['sub_category'] == item_b_attr['sub_category']
    
    return True

def validate_substitution(sales_df, oos_df, price_df, promo_df, item_a, item_b, control_vars=None):
    """
    Validate substitution relationship using statistical models
    
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
        Additional control variables
        
    Returns:
    --------
    dict
        Dictionary with validation results
    """
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
    
    # Add any additional control variables
    if control_vars is not None:
        analysis_df = pd.concat([analysis_df, control_vars], axis=1)
    
    # Drop rows with missing values
    analysis_df = analysis_df.dropna()
    
    if len(analysis_df) < 30:  # Not enough data for reliable analysis
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
        # First, run a linear model for OOS and promo effects
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
        
        # Add control variables if present
        if control_vars is not None:
            control_cols = [col for col in control_vars.columns if col in analysis_df.columns]
            if control_cols:
                linear_predictor_parts.extend(control_cols)
        
        # Create linear formula
        linear_formula = linear_formula_parts[0] + ' + '.join(linear_predictor_parts)
        
        # Run the linear model for OOS and promo effects
        model_linear = sm.formula.ols(formula=linear_formula, data=analysis_df).fit()
        
        # Now create the log-log model for price elasticity
        # Create log transformations for price elasticity
        analysis_df['log_sales_b'] = np.log(analysis_df['sales_b'].replace(0, 0.01))
        
        if 'price_a' in analysis_df.columns:
            analysis_df['log_price_a'] = np.log(analysis_df['price_a'])
        
        if 'price_b' in analysis_df.columns:
            analysis_df['log_price_b'] = np.log(analysis_df['price_b'])
        
        # Build log-log model formula
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
        
        # Add promo effects
        if 'promo_a' in analysis_df.columns:
            log_predictor_parts.append('promo_a')
        
        # Add control variables if present
        if control_vars is not None:
            control_cols = [col for col in control_vars.columns if col in analysis_df.columns]
            if control_cols:
                log_predictor_parts.extend(control_cols)
        
        # Create log-log formula
        log_formula = log_formula_parts[0] + ' + '.join(log_predictor_parts)
        
        # Run the log-log model for price elasticity
        model_log = sm.formula.ols(formula=log_formula, data=analysis_df).fit()
        
        # Extract effects and p-values
        oos_effect = model_linear.params.get('oos_a', 0) if 'oos_a' in model_linear.params else 0
        oos_pvalue = model_linear.pvalues.get('oos_a', 1) if 'oos_a' in model_linear.pvalues else 1
        
        promo_effect = model_linear.params.get('promo_a', 0) if 'promo_a' in model_linear.params else 0
        promo_pvalue = model_linear.pvalues.get('promo_a', 1) if 'promo_a' in model_linear.pvalues else 1
        
        price_effect = model_log.params.get('log_price_a', 0) if 'log_price_a' in model_log.params else 0
        price_pvalue = model_log.pvalues.get('log_price_a', 1) if 'log_price_a' in model_log.pvalues else 1
        
        # Set a maximum cap for relative effects
        MAX_EFFECT_CAP = 5.0
        
        # Check significance
        oos_significant = oos_pvalue < 0.05 and oos_effect > 0
        price_significant = price_pvalue < 0.05 and price_effect > 0
        promo_significant = promo_pvalue < 0.05 and promo_effect < 0
        
        # Calculate mean sales for relative effects
        mean_sales = analysis_df['sales_b'].mean()
        
        # Initialize effect values
        oos_relative_effect = 0
        promo_relative_effect = 0
        
        # Only calculate effects for statistically significant results
        if oos_significant and mean_sales > 0.01:
            oos_relative_effect = min(oos_effect / mean_sales, MAX_EFFECT_CAP)
        
        if promo_significant and mean_sales > 0.01:
            promo_relative_effect = max(-promo_effect / mean_sales, -MAX_EFFECT_CAP)
            
        # Only keep the price effect if significant
        if not price_significant:
            price_effect = 0
        else:
            price_effect = min(price_effect, MAX_EFFECT_CAP)
        
        # Store the results (only essential fields for CSV generation)
        results.update({
            'oos_effect': oos_relative_effect,
            'oos_significant': oos_significant,
            'promo_effect': promo_relative_effect,
            'promo_significant': promo_significant,
            'price_effect': price_effect,
            'price_significant': price_significant,
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

def add_relationship_type(result):
    """
    Add relationship_type field to results
    
    Parameters:
    -----------
    result : dict
        Result dictionary from validate_substitution
        
    Returns:
    --------
    dict
        The same result dictionary with relationship_type added
    """
    if not result.get('validation_successful', False):
        return result
    
    # Get effect values
    price_effect = result.get('price_effect', 0)
    oos_effect = result.get('oos_effect', 0)
    promo_effect = result.get('promo_effect', 0)
    
    # Overall relationship categorization
    if (result.get('price_significant', False) and price_effect > 0) or (result.get('oos_significant', False) and oos_effect > 0):
        result['relationship_type'] = "Substitute"
    elif (result.get('price_significant', False) and price_effect < 0) or (result.get('promo_significant', False) and promo_effect < 0):
        result['relationship_type'] = "Complement"
    else:
        result['relationship_type'] = "Undefined"
    
    return result

def calculate_substitution_effects(sales_df, oos_df, price_df, promo_df, items_list, 
                                  min_oos_days=5, control_vars=None, product_attributes=None, 
                                  substitution_scope="category"):
    """
    Calculate substitution effects using linear and log-log models
    
    Parameters:
    -----------
    sales_df : DataFrame
        Pivot table with dates as index and items as columns, values are sales
    oos_df : DataFrame
        Pivot table with OOS flags
    price_df : DataFrame
        Pivot table with prices
    promo_df : DataFrame
        Pivot table with promotion flags
    items_list : list
        List of item IDs to analyze
    min_oos_days : int
        Minimum number of OOS days required
    control_vars : DataFrame
        Control variables for regression
    product_attributes : DataFrame
        Product attributes data with category information
    substitution_scope : str
        Scope for substitution filtering: "category", "sub_category", or "all"
        
    Returns:
    --------
    tuple
        (detailed_results)
    """
    logger.info(f"Calculating substitution effects with combined model")
    
    detailed_results = {}
    
    item_count = len(items_list)
    processed = 0
    
    for item_a in items_list:
        processed += 1
        if processed % 20 == 0:
            logger.info(f"Substitution Analysis: Processed {processed}/{item_count} items")
        
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
            
            # Skip if not in same category/subcategory based on substitution scope
            if not check_substitution_scope(item_a, item_b, product_attributes, substitution_scope):
                continue
                
            # Use combined validation function
            result = validate_substitution(
                sales_df, oos_df, price_df, promo_df, 
                item_a, item_b, control_vars
            )
            
            # Add relationship type
            result = add_relationship_type(result)
            
            detailed_results[item_a][item_b] = result
    
    logger.info(f"Substitution analysis complete")
    return detailed_results

def find_substitutes(detailed_results, k=5, require_significance=True):
    """
    Identify top substitutes using combined model results
    
    Parameters:
    -----------
    detailed_results : dict
        Detailed results from calculate_substitution_effects
    k : int
        Number of top substitutes to return for each item
    require_significance : bool
        If True, require at least one significant effect
        
    Returns:
    --------
    dict
        Dictionary mapping items to their top substitutes
    """
    logger.info(f"Finding top {k} substitutes for each item")
    
    # Extract item list from results
    items_list = list(detailed_results.keys())
    
    # Create combined score matrix
    combined_scores = pd.DataFrame(0.0, index=items_list, columns=items_list)
    
    # Set maximum caps for effects
    MAX_OOS_EFFECT = 5.0
    MAX_PRICE_EFFECT = 3.0
    MAX_PROMO_EFFECT = 3.0
    
    # Process each item pair
    for item_a in items_list:
        for item_b in detailed_results.get(item_a, {}):
            result = detailed_results[item_a][item_b]
            
            # Skip if validation was not successful
            if not result.get('validation_successful', False):
                continue
            
            # Get effect values with capping
            oos_effect = min(result.get('oos_effect', 0), MAX_OOS_EFFECT) if result.get('oos_significant', False) else 0
            price_effect = min(result.get('price_effect', 0), MAX_PRICE_EFFECT) if result.get('price_significant', False) else 0
            promo_effect = min(abs(result.get('promo_effect', 0)), MAX_PROMO_EFFECT) if result.get('promo_significant', False) else 0
            
            # Count significant effects
            sig_count = sum([
                1 if result.get('oos_significant', False) and oos_effect > 0 else 0,
                1 if result.get('price_significant', False) and price_effect > 0 else 0,
                1 if result.get('promo_significant', False) and promo_effect < 0 else 0
            ])
            
            # Skip if require_significance is True and no significant effects
            if require_significance and sig_count == 0:
                continue
            
            # Calculate combined score with equal weights
            if sig_count > 0:
                # Normalize to give equal weight to each significant dimension
                weight = 1.0 / sig_count
                score = weight * (oos_effect + price_effect + abs(promo_effect))
                
                # Store the score
                combined_scores.loc[item_a, item_b] = score
    
    # Create substitutes dictionary
    substitutes_dict = {}
    
    for item_a in items_list:
        substitutes_dict[item_a] = []
        
        # Get scores for all items
        item_scores = combined_scores.loc[item_a].drop(item_a, errors='ignore')
        
        # Only include items with non-zero scores
        item_scores = item_scores[item_scores > 0]
        
        # Sort by combined score (descending)
        top_items = item_scores.sort_values(ascending=False).head(k)
        
        for item_b, score in top_items.items():
            details = detailed_results[item_a][item_b]
            substitutes_dict[item_a].append((item_b, score, details))
    
    # Calculate some stats
    items_with_substitutes = sum(1 for item, subs in substitutes_dict.items() if len(subs) > 0)
    total_substitutes = sum(len(subs) for subs in substitutes_dict.values())
    
    logger.info(f"Found substitutes for {items_with_substitutes} items, "
               f"total of {total_substitutes} substitute relationships")
                
    return substitutes_dict

def save_results(results, filepath):
    """
    Save results to pickle file
    
    Parameters:
    -----------
    results : object
        Results to save
    filepath : str
        Output filepath
    """
    try:
        ensure_dir(os.path.dirname(filepath))
        pd.to_pickle(results, filepath)
        logger.info(f"Results saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def export_to_csv(substitutes_dict, output_dir):
    """
    Export substitution results to CSV format
    
    Parameters:
    -----------
    substitutes_dict : dict
        Dictionary with substitution results
    output_dir : str
        Directory to save CSV file
        
    Returns:
    --------
    str
        Path to the CSV output file
    """
    logger.info(f"Exporting results to CSV")
    
    # Create output directory if needed
    ensure_dir(output_dir)
    
    # Extract substitutes to dataframe (minimal fields only)
    substitutes_list = []
    for item_a, subs in substitutes_dict.items():
        for item_b, score, details in subs:
            substitutes_list.append({
                'primary_item': item_a,
                'substitute_item': item_b,
                'combined_score': score,
                'oos_effect': details.get('oos_effect', 0),
                'price_effect': details.get('price_effect', 0),
                'promo_effect': details.get('promo_effect', 0),
                'relationship_type': details.get('relationship_type', 'unknown')
            })
    
    # Create and save dataframe
    subs_df = pd.DataFrame(substitutes_list)
    subs_csv_path = os.path.join(output_dir, 'substitutes.csv')
    subs_df.to_csv(subs_csv_path, index=False)
    logger.info(f"Saved substitutes to {subs_csv_path}")
    
    return subs_csv_path

def setup_logging(log_dir="logs", verbose=False):
    """
    Set up logging configuration
    
    Parameters:
    -----------
    log_dir : str
        Directory for log files
    verbose : bool
        If True, enable INFO level logging
    """
    ensure_dir(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"substitution_{timestamp}.log")
    
    # Set log level based on verbose flag
    log_level = logging.INFO if verbose else logging.WARNING
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def run_substitution_analysis(config_path, export_csv=True, verbose=False):
    """
    Run the minimal substitution analysis pipeline
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    export_csv : bool
        Whether to export results to CSV format
    verbose : bool
        Enable verbose logging if True
        
    Returns:
    --------
    dict
        Dictionary with substitution results
    """
    # Set up logging
    setup_logging(verbose=verbose)
    logger = logging.getLogger(__name__)
    logger.info("Starting minimal SKU Substitution Analysis")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create output directories
    ensure_dir(config['data']['results_dir'])
    
    # Load data
    logger.info("Loading transaction data")
    raw_transactions_df = load_transaction_data(config['data']['input_file'])
    
    logger.info("Loading product attributes")
    try:
        attributes_df = load_product_attributes(config['data']['attributes_file'])
    except FileNotFoundError:
        logger.warning("Product attributes file not found, continuing without attributes")
        attributes_df = None
    
    # Run preprocessing
    logger.info("Running preprocessing and validation")
    transactions_df, anomalies = validate_and_preprocess(raw_transactions_df)
    
    if anomalies:
        logger.warning(f"Detected {len(anomalies)} types of anomalies in the data")
        
    # Filter sparse items if configured
    if config.get('data', {}).get('filter_sparse_items', False):
        min_days = config.get('data', {}).get('min_days', 30)
        logger.info(f"Filtering sparse items (min_days={min_days})")
        transactions_df = filter_sparse_items(
            transactions_df, 
            min_days=min_days
        )
    
    # Get list of items to analyze
    items_list = transactions_df['item_id'].unique().tolist()
    logger.info(f"Analyzing {len(items_list)} unique SKUs")
    
    # Create features and pivots
    logger.info("Creating features and pivots")
    feature_set = create_feature_set(
        transactions_df, 
        items_list,
        baseline_window=config['price_analysis']['baseline_window'],
        min_periods=config['price_analysis']['min_baseline_periods']
    )
    
    # Extract required data for analysis
    sales_pivot = feature_set['sales_pivot']
    price_pivot = feature_set['price_pivot']
    promo_pivot = feature_set['promo_pivot']
    oos_pivot = feature_set['oos_pivot']
    control_pivot = feature_set['control_pivot']
    
    # Create enhanced control variables for regression analyses
    control_vars = create_control_variables(sales_pivot, control_pivot)
    
    # Get substitution scope from config
    substitution_scope = config.get('analysis', {}).get('substitution_scope', "sub_category")
    
    # Calculate all substitution effects
    logger.info("Calculating substitution effects")
    detailed_results = calculate_substitution_effects(
        sales_pivot, 
        oos_pivot, 
        price_pivot, 
        promo_pivot,
        items_list, 
        min_oos_days=config['analysis']['min_oos_days'],
        control_vars=control_vars,
        product_attributes=attributes_df,
        substitution_scope=substitution_scope
    )
    
    # Find top substitutes
    logger.info("Finding top substitutes")
    substitutes_dict = find_substitutes(
        detailed_results,
        k=config['analysis']['top_k'],
        require_significance=config['analysis']['require_significance']
    )
    
    # Save minimal results
    results_path = os.path.join(config['data']['results_dir'], 'substitution_results.pkl')
    logger.info(f"Saving substitution results to {results_path}")
    
    # Prepare minimal results dictionary
    results_dict = {
        'substitutes_dict': substitutes_dict,
        'detailed_results': detailed_results
    }
    
    # Save the results
    save_results(results_dict, results_path)
    
    # Export to CSV if requested
    csv_path = None
    if export_csv:
        logger.info("Exporting results to CSV")
        csv_output_dir = os.path.join(config['data']['results_dir'], 'csv')
        csv_path = export_to_csv(substitutes_dict, csv_output_dir)
        logger.info(f"Results exported to CSV: {csv_path}")
    
    # Print a summary of results
    summary = f"""
    ===== ANALYSIS SUMMARY =====
    Analyzed {len(items_list)} unique products
    Found substitution relationships for {len([k for k, v in substitutes_dict.items() if v])} products
    Total substitution relationships: {sum(len(v) for v in substitutes_dict.values())}
    
    Results saved to:
    - Pickle format: {results_path}
    """
    
    if csv_path:
        summary += f"- CSV format: {csv_path}\n"
    
    print(summary)
    
    return substitutes_dict