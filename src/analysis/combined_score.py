#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined scoring for substitute products using the combined model.
Improved to account for different scales of OOS, price elasticity, and promo effects.
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
    Identify top substitutes using the combined model results with dynamic weights based on significance
    and normalized effects to account for different scales.
    
    Parameters:
    -----------
    combined_results : dict
        Detailed results from the combined validation model
    k : int
        Number of top substitutes to return for each item
    weights : dict
        Base weights for different effects in combined score (oos, price, promo)
        These will be adjusted dynamically based on which effects are significant
    require_significance : bool
        If True, item pairs must have at least one significant effect to be considered
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
    
    # Collect all significant effect values to determine scale factors
    all_oos_effects = []
    all_price_effects = []
    all_promo_effects = []
    
    # First pass to collect effect values
    for item_a in items_list:
        for item_b in combined_results.get(item_a, {}):
            result = combined_results[item_a][item_b]
            
            # Skip if validation was not successful
            if not result.get('validation_successful', False):
                continue
            
            # Extract all effects
            oos_effect = result.get('oos_effect', 0)
            oos_significant = result.get('oos_significant', False)
            
            price_effect = result.get('price_effect', 0)
            price_significant = result.get('price_significant', False)
            
            promo_effect = abs(result.get('promo_effect', 0))  # Use absolute value as promo effect is negative
            promo_significant = result.get('promo_significant', False)
            
            # Collect significant effects
            if oos_significant and oos_effect > 0:
                all_oos_effects.append(oos_effect)
                
            if price_significant and price_effect > 0:
                all_price_effects.append(price_effect)
                
            if promo_significant and promo_effect > 0:
                all_promo_effects.append(promo_effect)
    
    # Calculate median values for each effect type (more robust than mean)
    # Default to 1.0 if no significant effects found
    median_oos = np.median(all_oos_effects) if all_oos_effects else 1.0
    median_price = np.median(all_price_effects) if all_price_effects else 1.0
    median_promo = np.median(all_promo_effects) if all_promo_effects else 1.0
    
    # Calculate normalization factors to bring effects to similar scales
    # We use the median ratios to determine how to scale each effect type
    
    # Get the average of medians across all effect types
    median_avg = np.mean([median_oos, median_price, median_promo])
    
    # Calculate scale factors to bring each effect type to similar scale
    scale_factors = {
        'oos': median_avg / median_oos if median_oos > 0 else 1.0,
        'price': median_avg / median_price if median_price > 0 else 1.0,
        'promo': median_avg / median_promo if median_promo > 0 else 1.0
    }
    
    logger.info(f"Effect scale factors: OOS={scale_factors['oos']:.3f}, "
               f"Price={scale_factors['price']:.3f}, Promo={scale_factors['promo']:.3f}")
    
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
            
            promo_effect = abs(result.get('promo_effect', 0))  # Use absolute value as promo effect is negative
            promo_significant = result.get('promo_significant', False)
            
            # Determine which effects are significant and positive
            significant_effects = {
                'oos': oos_significant and oos_effect > 0,
                'price': price_significant and price_effect > 0,
                'promo': promo_significant and promo_effect > 0
            }
            
            # Count number of significant effects
            sig_count = sum(significant_effects.values())
            
            # Skip if require_significance is True and no significant effects
            if require_significance and sig_count == 0:
                continue
            
            # Calculate combined score using only significant effects
            if sig_count > 0:
                # Calculate dynamic weights based on which effects are significant
                dynamic_weights = {}
                for effect_name in ['oos', 'price', 'promo']:
                    if significant_effects[effect_name]:
                        dynamic_weights[effect_name] = 1.0 / sig_count
                    else:
                        dynamic_weights[effect_name] = 0.0
                
                # Create effect dict with original effect values
                original_effects = {
                    'oos': oos_effect,
                    'price': price_effect,
                    'promo': promo_effect
                }
                
                # Scale effects to account for different scales
                scaled_effects = {
                    name: original_effects[name] * scale_factors[name] 
                    for name in ['oos', 'price', 'promo']
                }
                
                # Apply weights to scaled effects
                weighted_effects = {
                    name: scaled_effects[name] * dynamic_weights[name] 
                    for name in ['oos', 'price', 'promo']
                }
                
                # Simple sum of weighted effects
                score = sum(weighted_effects.values())
                
                # Store the score
                combined_scores.loc[item_a, item_b] = score
                
                # Store detailed results
                detailed_results[item_a][item_b] = {
                    'combined_score': score,
                    'significant_dimensions': sig_count,
                    'dynamic_weights': dynamic_weights,
                    'oos_effect': oos_effect,
                    'oos_significant': oos_significant,
                    'oos_scaled': scaled_effects['oos'],
                    'oos_weighted': weighted_effects['oos'],
                    'price_effect': price_effect,
                    'price_significant': price_significant,
                    'price_scaled': scaled_effects['price'],
                    'price_weighted': weighted_effects['price'],
                    'price_effect_type': 'elasticity',  # Price effect is elasticity in the combined model
                    'promo_effect': promo_effect,
                    'promo_significant': promo_significant,
                    'promo_scaled': scaled_effects['promo'],
                    'promo_weighted': weighted_effects['promo'],
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
                
                # Determine dominant factor (only consider significant ones)
                dominant_factor = max(
                    {k: v for k, v in weighted_effects.items() if v > 0},
                    key=weighted_effects.get
                )
                
                # Map effect names to user-friendly names
                factor_names = {
                    'oos': 'availability',
                    'price': 'elasticity',
                    'promo': 'promotion'
                }
                
                detailed_results[item_a][item_b]['dominant_factor'] = factor_names.get(dominant_factor, dominant_factor)
    
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