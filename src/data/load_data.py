#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for loading transaction and product data.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)

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