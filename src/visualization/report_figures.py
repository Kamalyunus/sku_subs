#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate figures for final reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

def generate_report_figures(substitutes_dict, oos_matrix, price_matrix, price_type, output_dir):
    """
    Generate figures for the final report
    
    Parameters:
    -----------
    substitutes_dict : dict
        Dictionary of substitutes
    oos_matrix : DataFrame
        OOS substitution effect matrix
    price_matrix : DataFrame
        Price effect matrix
    price_type : DataFrame
        Matrix indicating price effect types
    output_dir : str
        Output directory for figures
        
    Returns:
    --------
    list
        List of generated figure paths
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger.info(f"Generating report figures in {output_dir}")
    
    generated_figures = []
    
    # Figure 1: Distribution of substitution scores
    logger.info("Generating substitution score distribution figure")
    
    all_scores = []
    for item, substitutes in substitutes_dict.items():
        for _, score, _ in substitutes:
            all_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(all_scores, bins=20, kde=True)
    plt.title('Distribution of Substitution Scores')
    plt.xlabel('Substitution Score')
    plt.ylabel('Frequency')
    
    output_path = os.path.join(output_dir, 'substitution_score_distribution.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    generated_figures.append(output_path)
    
    # Figure 2: Dominant factor breakdown
    logger.info("Generating dominant factor breakdown figure")
    
    dominant_factors = []
    for item, substitutes in substitutes_dict.items():
        for _, _, details in substitutes:
            if 'dominant_factor' in details:
                dominant_factors.append(details['dominant_factor'])
    
    factor_counts = pd.Series(dominant_factors).value_counts()
    
    # Ensure index is categorical to avoid matplotlib warning
    factor_counts.index = factor_counts.index.astype('category')
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=factor_counts.index, y=factor_counts.values)
    
    # Add percentages on top of bars
    total = factor_counts.sum()
    for i, count in enumerate(factor_counts):
        ax.text(i, count + 0.1, f'{count/total:.1%}', ha='center')
    
    plt.title('Dominant Factors in Substitution Relationships')
    plt.xlabel('Dominant Factor')
    plt.ylabel('Count')
    
    output_path = os.path.join(output_dir, 'dominant_factor_breakdown.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    generated_figures.append(output_path)
    
    # Add figure showing category matches in substitution relationships
    logger.info("Generating category analysis figure")
    
    # Check if category information is available in substitutes
    has_category_info = False
    for item, substitutes in substitutes_dict.items():
        if substitutes:
            if 'category_a' in substitutes[0][2] and 'category_b' in substitutes[0][2]:
                has_category_info = True
                break
    
    if has_category_info:
        # Count same category vs different category substitutions
        same_category = 0
        diff_category = 0
        category_pairs = []
        
        for item, substitutes in substitutes_dict.items():
            for _, score, details in substitutes:
                if 'category_a' in details and 'category_b' in details:
                    if details['category_a'] == details['category_b']:
                        same_category += 1
                        category_pairs.append(('Same Category', score))
                    else:
                        diff_category += 1
                        category_pairs.append(('Different Category', score))
        
        # Create figure showing category relationships
        plt.figure(figsize=(10, 6))
        
        # Simple bar chart of same vs different
        category_counts = pd.Series({'Same Category': same_category, 'Different Category': diff_category})
        ax = sns.barplot(x=category_counts.index, y=category_counts.values)
        
        # Add percentages on top of bars
        total = category_counts.sum()
        for i, count in enumerate(category_counts):
            ax.text(i, count + 0.1, f'{count/total:.1%}', ha='center')
        
        plt.title('Substitution Relationships by Category')
        plt.ylabel('Count')
        
        output_path = os.path.join(output_dir, 'category_substitution_analysis.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        generated_figures.append(output_path)
        
        # Boxplot of substitution scores by category relationship
        if category_pairs:
            plt.figure(figsize=(10, 6))
            category_df = pd.DataFrame(category_pairs, columns=['Category Match', 'Substitution Score'])
            sns.boxplot(data=category_df, x='Category Match', y='Substitution Score')
            plt.title('Substitution Scores by Category Relationship')
            
            output_path = os.path.join(output_dir, 'category_score_distribution.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            generated_figures.append(output_path)
    
    # Figure 3: Heatmap of top OOS substitution effects
    logger.info("Generating OOS heatmap figure")
    
    # Get top items by row sum of OOS effects
    top_oos_items = oos_matrix.sum(axis=1).sort_values(ascending=False).head(15).index
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(oos_matrix.loc[top_oos_items, top_oos_items], 
                annot=False, cmap='YlGnBu', cbar=True)
    plt.title('OOS Substitution Effect Heatmap (Top 15 Items)')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'oos_heatmap.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    generated_figures.append(output_path)
    
    # Figure 4: Price effect type breakdown
    logger.info("Generating price effect type figure")
    
    # Count by price effect type
    price_type_flat = price_type.values.flatten()
    price_type_counts = pd.Series(price_type_flat).value_counts()
    
    # Remove 'none' category if present
    if 'none' in price_type_counts:
        price_type_counts = price_type_counts.drop('none')
    
    # Ensure index is categorical to avoid matplotlib warning
    price_type_counts.index = price_type_counts.index.astype('category')
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=price_type_counts.index, y=price_type_counts.values)
    
    # Add percentages on top of bars
    total = price_type_counts.sum()
    for i, count in enumerate(price_type_counts):
        ax.text(i, count + 0.1, f'{count/total:.1%}', ha='center')
    
    plt.title('Price Effect Types in Substitution Relationships')
    plt.xlabel('Price Effect Type')
    plt.ylabel('Count')
    
    output_path = os.path.join(output_dir, 'price_effect_types.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    generated_figures.append(output_path)
    
    # Figure 5: Substitution network degree distribution
    logger.info("Generating degree distribution figure")
    
    # Count substitutes per item
    substitutes_per_item = {item: len(subs) for item, subs in substitutes_dict.items()}
    degree_counts = pd.Series(substitutes_per_item).value_counts().sort_index()
    
    # Convert index to categorical explicitly to avoid matplotlib warning
    degree_counts.index = degree_counts.index.astype('str')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=degree_counts.index, y=degree_counts.values)
    plt.title('Number of Substitutes per Item')
    plt.xlabel('Number of Substitutes')
    plt.ylabel('Count of Items')
    
    output_path = os.path.join(output_dir, 'substitutes_per_item.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    generated_figures.append(output_path)
    
    logger.info(f"Generated {len(generated_figures)} report figures")
    return generated_figures

def generate_comparative_analysis(substitutes_dict, transactions_df, product_attributes=None, output_dir=None):
    """
    Generate comparative analysis between substitutes
    
    Parameters:
    -----------
    substitutes_dict : dict
        Dictionary of substitutes
    transactions_df : DataFrame
        Transaction data
    product_attributes : DataFrame
        Product attributes data
    output_dir : str
        Output directory for figures
        
    Returns:
    --------
    tuple
        (DataFrame with comparative analysis, list of figure paths)
    """
    logger.info("Generating comparative analysis")
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Calculate average sales and price by item
    item_stats = transactions_df.groupby('item_id').agg({
        'sales': 'mean',
        'price': 'mean',
        'is_on_promotion': 'mean',
        'is_out_of_stock': 'mean'
    }).rename(columns={
        'is_on_promotion': 'promo_frequency',
        'is_out_of_stock': 'oos_frequency'
    })
    
    # Create comparison dataframe
    comparisons = []
    
    for item_a, substitutes in substitutes_dict.items():
        if item_a not in item_stats.index:
            continue
            
        item_a_stats = item_stats.loc[item_a]
        
        for item_b, score, details in substitutes:
            if item_b not in item_stats.index:
                continue
                
            item_b_stats = item_stats.loc[item_b]
            
            # Calculate relative differences
            price_diff = (item_b_stats['price'] / item_a_stats['price'] - 1) * 100
            sales_diff = (item_b_stats['sales'] / item_a_stats['sales'] - 1) * 100
            
            # Add category info if available
            category_match = False
            subcategory_match = False
            
            if product_attributes is not None:
                if item_a in product_attributes.set_index('item_id').index and \
                   item_b in product_attributes.set_index('item_id').index:
                    
                    attr_a = product_attributes.set_index('item_id').loc[item_a]
                    attr_b = product_attributes.set_index('item_id').loc[item_b]
                    
                    if 'category' in attr_a and 'category' in attr_b:
                        category_match = attr_a['category'] == attr_b['category']
                        
                    if 'sub_category' in attr_a and 'sub_category' in attr_b:
                        subcategory_match = attr_a['sub_category'] == attr_b['sub_category']
            
            comparisons.append({
                'item_a': item_a,
                'item_b': item_b,
                'substitution_score': score,
                'dominant_factor': details.get('dominant_factor', 'unknown'),
                'price_a': item_a_stats['price'],
                'price_b': item_b_stats['price'],
                'price_diff_pct': price_diff,
                'sales_a': item_a_stats['sales'],
                'sales_b': item_b_stats['sales'],
                'sales_diff_pct': sales_diff,
                'promo_freq_a': item_a_stats['promo_frequency'],
                'promo_freq_b': item_b_stats['promo_frequency'],
                'oos_freq_a': item_a_stats['oos_frequency'],
                'oos_freq_b': item_b_stats['oos_frequency'],
                'category_match': category_match,
                'subcategory_match': subcategory_match
            })
    
    comp_df = pd.DataFrame(comparisons)
    
    # Generate figures if output_dir provided
    generated_figures = []
    
    if output_dir:
        # Figure 1: Price difference vs. substitution score
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            data=comp_df,
            x='price_diff_pct',
            y='substitution_score',
            hue='dominant_factor',
            alpha=0.7
        )
        plt.title('Price Difference vs. Substitution Score')
        plt.xlabel('Price Difference (%)')
        plt.ylabel('Substitution Score')
        plt.axvline(x=0, color='gray', linestyle='--')
        
        output_path = os.path.join(output_dir, 'price_diff_vs_score.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        generated_figures.append(output_path)
        
        # Figure 2: Category match analysis
        if 'category_match' in comp_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(
                data=comp_df,
                x='category_match',
                y='substitution_score'
            )
            plt.title('Substitution Score by Category Match')
            plt.xlabel('Same Category')
            plt.ylabel('Substitution Score')
            
            output_path = os.path.join(output_dir, 'category_match_analysis.png')
            plt.savefig(output_path, dpi=300)
            plt.close()
            
            generated_figures.append(output_path)
    
    logger.info(f"Comparative analysis complete with {len(comp_df)} substitution pairs")
    
    return comp_df, generated_figures