#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined scoring for substitute products using the combined model.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def find_substitutes_with_validation(combined_results, k=5, 
                                    weights={'oos': 0.33, 'price': 0.33, 'promo': 0.33},
                                    require_significance=True, product_attributes=None,
                                    substitution_scope="category"):
    """
    Identify top substitutes using the combined model results
    
    Parameters:
    -----------
    combined_results : dict
        Detailed results from the combined validation model
    k : int
        Number of top substitutes to return for each item
    weights : dict
        Weights for different effects in combined score (oos, price, promo)
    require_significance : bool
        If True, only consider effects that are statistically significant
    product_attributes : DataFrame, optional
        Product attributes data with category information
    substitution_scope : str, default="category"
        Scope for finding substitutes: "category" (within same category only), 
        "sub_category" (within same sub-category only), or "all" (no restrictions)
        
    Returns:
    --------
    tuple
        (substitutes_dict, combined_scores, detailed_results)
    """
    logger.info(f"Finding top {k} substitutes with combined model results")
    
    # Extract item list from results
    items_list = list(combined_results.keys())
    
    # Create combined score matrix
    combined_scores = pd.DataFrame(0.0, index=items_list, columns=items_list)
    detailed_results = {}
    
    # Create item category mapping if product attributes are available
    item_categories = {}
    item_subcategories = {}
    
    if product_attributes is not None and substitution_scope in ["category", "sub_category"]:
        logger.info(f"Applying substitution scope: {substitution_scope}")
        # Create mapping of item to category/sub_category
        if 'item_id' in product_attributes.columns and 'category' in product_attributes.columns:
            item_categories = dict(zip(product_attributes['item_id'], product_attributes['category']))
            
        if 'item_id' in product_attributes.columns and 'sub_category' in product_attributes.columns:
            item_subcategories = dict(zip(product_attributes['item_id'], product_attributes['sub_category']))
    
    # Process each item pair
    for item_a in items_list:
        detailed_results[item_a] = {}
        
        for item_b in items_list:
            if item_a == item_b:
                continue
                
            # Skip if no results
            if item_b not in combined_results.get(item_a, {}):
                continue
                
            # Apply category/sub-category restriction if needed
            if substitution_scope == "category" and item_categories:
                # Skip if items are not in the same category
                if item_a not in item_categories or item_b not in item_categories:
                    continue
                if item_categories[item_a] != item_categories[item_b]:
                    continue
            
            elif substitution_scope == "sub_category" and item_subcategories:
                # Skip if items are not in the same sub-category
                if item_a not in item_subcategories or item_b not in item_subcategories:
                    continue
                if item_subcategories[item_a] != item_subcategories[item_b]:
                    continue
                
            # Get combined model results
            result = combined_results[item_a][item_b]
            
            # Skip if validation was not successful
            if not result.get('validation_successful', False):
                continue
            
            # Extract all effects
            oos_effect = result.get('oos_effect', 0)
            oos_significant = result.get('oos_significant', False)
            
            price_effect = result.get('price_effect', 0)
            price_significant = result.get('price_significant', False)
            
            promo_effect = abs(result.get('promo_effect', 0))  # Use absolute value as promo effect is negative for cannibalization
            promo_significant = result.get('promo_significant', False)
            
            # Apply significance filter
            if require_significance:
                if not oos_significant:
                    oos_effect = 0
                if not price_significant:
                    price_effect = 0
                if not promo_significant:
                    promo_effect = 0
                
            # Calculate combined score
            if oos_effect > 0 or price_effect > 0 or promo_effect > 0:
                # Normalize effects
                effects = [oos_effect, price_effect, promo_effect]
                max_effect = max(effects)
                
                if max_effect > 0:
                    oos_norm = oos_effect / max_effect
                    price_norm = price_effect / max_effect
                    promo_norm = promo_effect / max_effect
                    
                    score = (
                        weights.get('oos', 0.33) * oos_norm + 
                        weights.get('price', 0.33) * price_norm + 
                        weights.get('promo', 0.33) * promo_norm
                    )
                    
                    combined_scores.loc[item_a, item_b] = score
                    
                    # Store detailed results
                    detailed_results[item_a][item_b] = {
                        'combined_score': score,
                        'oos_effect': oos_effect,
                        'oos_significant': oos_significant,
                        'price_effect': price_effect,
                        'price_significant': price_significant,
                        'price_effect_type': 'elasticity',  # Price effect is elasticity in the combined model
                        'promo_effect': promo_effect,
                        'promo_significant': promo_significant,
                        'linear_model_r2': result.get('linear_model_r2', 0),
                        'log_model_r2': result.get('log_model_r2', 0)
                    }
                    
                    # Add category information if available
                    if item_a in item_categories:
                        detailed_results[item_a][item_b]['category_a'] = item_categories[item_a]
                    if item_b in item_categories:
                        detailed_results[item_a][item_b]['category_b'] = item_categories[item_b]
                    if item_a in item_subcategories:
                        detailed_results[item_a][item_b]['sub_category_a'] = item_subcategories[item_a]
                    if item_b in item_subcategories:
                        detailed_results[item_a][item_b]['sub_category_b'] = item_subcategories[item_b]
                    
                    # Determine dominant factor
                    oos_contrib = oos_effect * weights.get('oos', 0.33)
                    price_contrib = price_effect * weights.get('price', 0.33)
                    promo_contrib = promo_effect * weights.get('promo', 0.33)
                    
                    contributions = {
                        'availability': oos_contrib,
                        'elasticity': price_contrib,
                        'promotion': promo_contrib
                    }
                    
                    dominant_factor = max(contributions, key=contributions.get)
                    detailed_results[item_a][item_b]['dominant_factor'] = dominant_factor
    
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
                
    return substitutes_dict, combined_scores, detailed_results