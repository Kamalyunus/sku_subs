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
    Identify top substitutes using the combined model results with dynamic weights based on significance
    
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
            
            # Determine which effects are significant
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
                
                # Normalize effects
                effects = {
                    'oos': oos_effect,
                    'price': price_effect,
                    'promo': promo_effect
                }
                
                # Get effects that are significant
                sig_effects = {k: v for k, v in effects.items() if significant_effects[k]}
                
                # If any effects are significant, proceed with combined score calculation
                if sig_effects:
                    # Find maximum effect value among significant effects
                    max_effect = max(sig_effects.values())
                    
                    if max_effect > 0:
                        # Normalize each effect relative to the maximum significant effect
                        oos_norm = oos_effect / max_effect if significant_effects['oos'] else 0
                        price_norm = price_effect / max_effect if significant_effects['price'] else 0
                        promo_norm = promo_effect / max_effect if significant_effects['promo'] else 0
                        
                        # Calculate combined score using dynamic weights
                        score = (
                            dynamic_weights['oos'] * oos_norm + 
                            dynamic_weights['price'] * price_norm + 
                            dynamic_weights['promo'] * promo_norm
                        )
                        
                        combined_scores.loc[item_a, item_b] = score
                        
                        # Store detailed results
                        detailed_results[item_a][item_b] = {
                            'combined_score': score,
                            'significant_dimensions': sig_count,
                            'dynamic_weights': dynamic_weights,
                            'oos_effect': oos_effect,
                            'oos_significant': oos_significant,
                            'oos_normalized': oos_norm,
                            'price_effect': price_effect,
                            'price_significant': price_significant,
                            'price_normalized': price_norm,
                            'price_effect_type': 'elasticity',  # Price effect is elasticity in the combined model
                            'promo_effect': promo_effect,
                            'promo_significant': promo_significant,
                            'promo_normalized': promo_norm,
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
                        sig_contributions = {
                            'availability': oos_effect * dynamic_weights['oos'] if significant_effects['oos'] else 0,
                            'elasticity': price_effect * dynamic_weights['price'] if significant_effects['price'] else 0,
                            'promotion': promo_effect * dynamic_weights['promo'] if significant_effects['promo'] else 0
                        }
                        
                        dominant_factor = max(sig_contributions, key=sig_contributions.get)
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