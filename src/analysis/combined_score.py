#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Combined scoring for substitute products.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def find_top_substitutes(oos_matrix, oos_significance, 
                         price_matrix, price_type, price_significance,
                         k=5, weights={'oos': 0.5, 'price': 0.5}, 
                         require_significance=True, elasticity_data=None,
                         product_attributes=None, substitution_scope="category"):
    """
    Combine OOS and price effects to find top k substitutes
    
    Parameters:
    -----------
    oos_matrix : DataFrame
        OOS substitution effect matrix
    oos_significance : DataFrame
        Boolean matrix indicating statistical significance of OOS effects
    price_matrix : DataFrame
        Combined price effect matrix (promotion + price matching) or elasticity matrix
    price_type : DataFrame
        Matrix indicating the type of price effect ("promotion", "price_matching", "substitute", "complement")
    price_significance : DataFrame
        Boolean matrix indicating statistical significance of price effects
    k : int
        Number of top substitutes to return for each item
    weights : dict
        Weights for different effects in combined score
    require_significance : bool
        If True, only consider effects that are statistically significant
    elasticity_data : dict, optional
        Additional elasticity data if available
    product_attributes : DataFrame, optional
        Product attributes data with category information
    substitution_scope : str, default="category"
        Scope for finding substitutes: "category" (within same category only), 
        "sub_category" (within same sub-category only), or "all" (no restrictions)
        
    Returns:
    --------
    tuple
        (substitutes_dict, combined_matrix)
    """
    logger.info(f"Finding top {k} substitutes with weights: {weights}")
    
    # Apply significance filter if required
    if require_significance:
        oos_filtered = oos_matrix.copy()
        price_filtered = price_matrix.copy()
        
        # Zero out non-significant effects
        oos_filtered[~oos_significance] = 0
        price_filtered[~price_significance] = 0
        
        logger.info(f"Applied significance filter: {oos_filtered.astype(bool).sum().sum()} significant OOS effects, "
                   f"{price_filtered.astype(bool).sum().sum()} significant price effects")
    else:
        oos_filtered = oos_matrix
        price_filtered = price_matrix
    
    # Normalize matrices to 0-1 scale
    oos_max = oos_filtered.max().max()
    price_max = price_filtered.max().max()
    
    if oos_max > 0:
        oos_norm = oos_filtered / oos_max
    else:
        oos_norm = oos_filtered
        
    if price_max > 0:
        price_norm = price_filtered / price_max
    else:
        price_norm = price_filtered
    
    # Handle elasticity values appropriately - substitute relationships have positive elasticities
    # If price_filtered contains elasticity values, we need to ensure positive values indicate substitutes
    is_elasticity = "substitute" in price_type.values
    if is_elasticity:
        logger.info("Using elasticity values in combined score")
        
        # For elasticity, we only want positive values (substitutes) in the combined score
        # Zero out negative values (complements)
        price_norm[price_norm < 0] = 0
    
    # Combined score
    combined_matrix = weights['oos'] * oos_norm + weights['price'] * price_norm
    
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
    
    # For each item, find top k substitutes
    substitutes_dict = {}
    
    for item_a in combined_matrix.index:
        substitutes_dict[item_a] = []
        
        # Get scores for all items except the item itself
        item_scores = combined_matrix.loc[item_a].drop(item_a, errors='ignore')
        
        # Only include items with non-zero scores
        item_scores = item_scores[item_scores > 0]
        
        # Apply category/sub-category restriction if needed
        if substitution_scope == "category" and item_categories and item_a in item_categories:
            # Filter to only include items in the same category
            same_category_items = [item for item in item_scores.index 
                                   if item in item_categories and 
                                   item_categories[item] == item_categories[item_a]]
            item_scores = item_scores.loc[same_category_items]
            logger.debug(f"Item {item_a}: Filtered to {len(item_scores)} items in same category {item_categories[item_a]}")
            
        elif substitution_scope == "sub_category" and item_subcategories and item_a in item_subcategories:
            # Filter to only include items in the same sub-category
            same_subcategory_items = [item for item in item_scores.index 
                                     if item in item_subcategories and 
                                     item_subcategories[item] == item_subcategories[item_a]]
            item_scores = item_scores.loc[same_subcategory_items]
            logger.debug(f"Item {item_a}: Filtered to {len(item_scores)} items in same sub-category {item_subcategories[item_a]}")
        
        # Sort by combined score (descending)
        top_items = item_scores.sort_values(ascending=False).head(k)
        
        for item_b, score in top_items.items():
            # Get details about each effect
            details = {
                'combined_score': score,
                'oos_effect': oos_filtered.loc[item_a, item_b],
                'oos_significant': bool(oos_significance.loc[item_a, item_b]),
                'price_effect': price_filtered.loc[item_a, item_b],
                'price_effect_type': price_type.loc[item_a, item_b],
                'price_significant': bool(price_significance.loc[item_a, item_b])
            }
            
            # Add category information if available
            if item_a in item_categories:
                details['category_a'] = item_categories[item_a]
            if item_b in item_categories:
                details['category_b'] = item_categories[item_b]
            if item_a in item_subcategories:
                details['sub_category_a'] = item_subcategories[item_a]
            if item_b in item_subcategories:
                details['sub_category_b'] = item_subcategories[item_b]
            
            # Include elasticity data if available
            if elasticity_data is not None and item_a in elasticity_data and item_b in elasticity_data[item_a]:
                details['elasticity'] = elasticity_data[item_a][item_b].get('elasticity', 0)
                details['elasticity_significant'] = elasticity_data[item_a][item_b].get('significant', False)
                details['elasticity_r_squared'] = elasticity_data[item_a][item_b].get('r_squared', 0)
            
            # Determine dominant factor
            oos_contrib = details['oos_effect'] * weights['oos']
            price_contrib = details['price_effect'] * weights['price']
            
            if oos_contrib > price_contrib:
                details['dominant_factor'] = "availability"
            else:
                # Use appropriate dominant factor based on the type of price effect
                if is_elasticity:
                    if details['price_effect_type'] in ['substitute', 'complement']:
                        details['dominant_factor'] = f"elasticity_{details['price_effect_type']}"
                    else:
                        details['dominant_factor'] = "elasticity"
                else:
                    details['dominant_factor'] = details['price_effect_type']
            
            substitutes_dict[item_a].append((item_b, score, details))
    
    # Calculate some stats
    items_with_substitutes = sum(1 for item, subs in substitutes_dict.items() if len(subs) > 0)
    total_substitutes = sum(len(subs) for subs in substitutes_dict.values())
    
    logger.info(f"Found substitutes for {items_with_substitutes} items, "
               f"total of {total_substitutes} substitute relationships")
                
    return substitutes_dict, combined_matrix

def find_substitutes_with_validation(oos_results, price_results, elasticity_results=None,
                                    k=5, weights={'oos': 0.5, 'price': 0.5},
                                    require_significance=True, product_attributes=None,
                                    substitution_scope="category"):
    """
    Enhanced substitute finding using the full validation results
    
    Parameters:
    -----------
    oos_results : dict
        Detailed OOS validation results from calculate_oos_substitution_with_validation
    price_results : dict
        Detailed price analysis results
    elasticity_results : dict, optional
        Detailed elasticity results if available
    k : int
        Number of top substitutes to return for each item
    weights : dict
        Weights for different effects in combined score
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
    logger.info(f"Finding top {k} substitutes with enhanced validation results")
    
    # Extract item list from results
    items_list = list(oos_results.keys())
    
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
                
            # Skip if no OOS results
            if item_b not in oos_results.get(item_a, {}):
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
                
            # Get OOS and price effects
            oos_effect = oos_results[item_a][item_b].get('oos_effect', 0)
            oos_significant = oos_results[item_a][item_b].get('oos_significant', False)
            
            price_effect = 0
            price_significant = False
            price_effect_type = "none"
            
            # Include price effects if available
            if price_results and item_a in price_results and item_b in price_results[item_a]:
                price_effect = price_results[item_a][item_b].get('price_effect', 0)
                price_significant = price_results[item_a][item_b].get('price_significant', False)
                price_effect_type = price_results[item_a][item_b].get('price_effect_type', "none")
            
            # Include elasticity if available
            elasticity = 0
            elasticity_significant = False
            
            if elasticity_results and item_a in elasticity_results and item_b in elasticity_results[item_a]:
                elasticity = elasticity_results[item_a][item_b].get('elasticity', 0)
                elasticity_significant = elasticity_results[item_a][item_b].get('significant', False)
            
            # Apply significance filter
            if require_significance:
                if not oos_significant:
                    oos_effect = 0
                if not price_significant:
                    price_effect = 0
                if not elasticity_significant:
                    elasticity = 0
            
            # Replace price effect with elasticity if available and requested
            if elasticity_results and elasticity > 0:
                price_effect = elasticity
                price_effect_type = "elasticity"
                
            # Calculate combined score
            if oos_effect > 0 or price_effect > 0:
                # Normalize within the pair
                max_effects = max(oos_effect, price_effect)
                if max_effects > 0:
                    oos_norm = oos_effect / max_effects
                    price_norm = price_effect / max_effects
                    
                    score = weights['oos'] * oos_norm + weights['price'] * price_norm
                    combined_scores.loc[item_a, item_b] = score
                    
                    # Store detailed results
                    detailed_results[item_a][item_b] = {
                        'combined_score': score,
                        'oos_effect': oos_effect,
                        'oos_significant': oos_significant,
                        'price_effect': price_effect,
                        'price_significant': price_significant,
                        'price_effect_type': price_effect_type,
                        'elasticity': elasticity,
                        'elasticity_significant': elasticity_significant
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
                    oos_contrib = oos_effect * weights['oos']
                    price_contrib = price_effect * weights['price']
                    
                    if oos_contrib > price_contrib:
                        detailed_results[item_a][item_b]['dominant_factor'] = "availability"
                    else:
                        detailed_results[item_a][item_b]['dominant_factor'] = price_effect_type
    
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