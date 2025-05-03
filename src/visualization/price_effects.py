#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualizations for price effects between substitute products.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging

logger = logging.getLogger(__name__)

def visualize_price_effects(item_a, item_b, transactions_df, price_change_types, discount_df, output_path=None):
    """
    Visualize price effects between two items
    
    Parameters:
    -----------
    item_a, item_b : str
        Item IDs to analyze
    transactions_df : DataFrame
        Transaction data
    price_change_types : DataFrame
        Price change categorization
    discount_df : DataFrame
        Discount percentages from baseline
    output_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure
        Figure with visualization
    """
    logger.info(f"Creating price effect visualization for {item_a} -> {item_b}")
    
    # Use transaction data directly as it's already at date-item level
    # Get data for the two items
    df_a = transactions_df[transactions_df['item_id'] == item_a].set_index('date')[['sales', 'price']]
    df_b = transactions_df[transactions_df['item_id'] == item_b].set_index('date')[['sales', 'price']]
    
    # Skip if we don't have data for both items
    if df_a.empty or df_b.empty:
        logger.warning(f"Missing data for one or both items: {item_a}, {item_b}")
        return None
    
    # Merge data
    merged = pd.DataFrame({
        'sales_a': df_a['sales'],
        'price_a': df_a['price'],
        'sales_b': df_b['sales'],
        'price_b': df_b['price']
    })
    
    # Add price change types if available
    if item_a in price_change_types.columns:
        merged['price_change_type'] = price_change_types[item_a]
    else:
        merged['price_change_type'] = 'unknown'
        
    # Add discount percentage if available
    if item_a in discount_df.columns:
        merged['discount_pct'] = discount_df[item_a]
    
    # Drop rows with missing values
    merged = merged.dropna()
    
    if len(merged) < 10:
        logger.warning(f"Not enough data points for visualization: {item_a}, {item_b}")
        return None
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    
    # Convert index to datetime if not already
    if not isinstance(merged.index, pd.DatetimeIndex):
        merged.index = pd.to_datetime(merged.index)
    
    # Format x-axis to show dates correctly
    import matplotlib.dates as mdates
    date_format = mdates.DateFormatter('%Y-%m-%d')
    
    # 1. Sales of both items
    ax1 = axes[0]
    ax1.set_title(f"Sales Comparison: {item_a} vs {item_b}")
    
    # Plot sales
    ax1.plot(merged.index, merged['sales_a'], color='blue', label=f"{item_a} Sales")
    ax1.set_ylabel(f"{item_a} Sales", color='blue')
    ax1.xaxis.set_major_formatter(date_format)
    
    # Create second y-axis
    ax1_2 = ax1.twinx()
    ax1_2.plot(merged.index, merged['sales_b'], color='red', label=f"{item_b} Sales")
    ax1_2.set_ylabel(f"{item_b} Sales", color='red')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 2. Price of item A and sales of item B
    ax2 = axes[1]
    ax2.set_title(f"Price of {item_a} vs Sales of {item_b}")
    
    # Plot price A
    ax2.plot(merged.index, merged['price_a'], color='blue', label=f"{item_a} Price")
    ax2.set_ylabel(f"{item_a} Price", color='blue')
    ax2.xaxis.set_major_formatter(date_format)
    
    # Create second y-axis
    ax2_2 = ax2.twinx()
    ax2_2.plot(merged.index, merged['sales_b'], color='red', label=f"{item_b} Sales")
    ax2_2.set_ylabel(f"{item_b} Sales", color='red')
    
    # Add legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Add x-axis label for better clarity
    plt.xlabel('Date')
    
    # 3. Scatter plot of price A vs sales B, colored by price change type
    ax3 = axes[2]
    ax3.set_title(f"Effect of {item_a} Price on {item_b} Sales")
    
    # Create color mapping
    colors = {
        'normal': 'gray',
        'discount': 'orange',
        'promotion': 'green',
        'unknown': 'blue'
    }
    
    # Create scatter plot for each price change type
    for change_type, color in colors.items():
        mask = merged['price_change_type'] == change_type
        if mask.sum() > 0:
            ax3.scatter(
                merged.loc[mask, 'price_a'], 
                merged.loc[mask, 'sales_b'],
                color=color,
                alpha=0.7,
                label=change_type
            )
    
    # Add regression line for overall trend
    sns.regplot(
        x='price_a', 
        y='sales_b', 
        data=merged,
        scatter=False,
        ci=None,
        line_kws={'color': 'black', 'linestyle': '--'},
        ax=ax3
    )
    
    # Add separate regression lines for each price change type
    for change_type, color in colors.items():
        mask = merged['price_change_type'] == change_type
        if mask.sum() > 10:  # Only if enough data points
            sns.regplot(
                x='price_a', 
                y='sales_b', 
                data=merged[mask],
                scatter=False,
                ci=None,
                line_kws={'color': color},
                ax=ax3
            )
    
    ax3.set_xlabel(f"{item_a} Price")
    ax3.set_ylabel(f"{item_b} Sales")
    ax3.legend()
    
    # Get actual date range from data
    if not merged.empty and isinstance(merged.index, pd.DatetimeIndex):
        # Use actual data range with a small buffer
        min_date = merged.index.min() - pd.Timedelta(days=1)
        max_date = merged.index.max() + pd.Timedelta(days=1)
    else:
        # Fallback to default 2024 range if no data
        min_date = pd.to_datetime('2024-01-01')
        max_date = pd.to_datetime('2024-12-31')
    
    for ax in axes[:2]:  # Apply only to time series plots
        # Set x-axis limits based on actual data
        ax.set_xlim([min_date, max_date])
        
        # Calculate appropriate interval based on date range
        days_span = (max_date - min_date).days
        if days_span <= 60:
            # For short ranges, show weekly ticks
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            date_format = '%d %b'
        elif days_span <= 180:
            # For medium ranges, show bi-weekly ticks
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=14))
            date_format = '%d %b'
        else:
            # For full year, show monthly ticks
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            date_format = '%b %Y'
        
        # Set formatter with appropriate date format
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        
        # Rotate date labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Ensure all dates are visible
        fig.autofmt_xdate()
        
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved price effect visualization to {output_path}")
    
    return fig

def visualize_top_pairs(substitutes_dict, transactions_df, price_change_types, discount_df, output_dir, max_pairs=10):
    """
    Create price effect visualizations for top substitute pairs
    
    Parameters:
    -----------
    substitutes_dict : dict
        Dictionary of substitutes
    transactions_df : DataFrame
        Transaction data
    price_change_types : DataFrame
        Price change categorization
    discount_df : DataFrame
        Discount percentages
    output_dir : str
        Directory to save visualizations
    max_pairs : int
        Maximum number of pairs to visualize
        
    Returns:
    --------
    list
        Paths to generated visualizations
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    logger.info(f"Creating price effect visualizations for top {max_pairs} substitute pairs")
    
    # Get top pairs by combined score
    all_pairs = []
    for item_a, substitutes in substitutes_dict.items():
        for item_b, score, details in substitutes:
            all_pairs.append((item_a, item_b, score))
    
    # Sort by score and take top pairs
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    top_pairs = all_pairs[:max_pairs]
    
    output_paths = []
    for item_a, item_b, score in top_pairs:
        output_path = os.path.join(output_dir, f"price_effect_{item_a}_{item_b}.png")
        
        # Create visualization
        fig = visualize_price_effects(
            item_a, item_b, 
            transactions_df, 
            price_change_types, 
            discount_df,
            output_path
        )
        
        if fig:
            output_paths.append(output_path)
            plt.close(fig)
    
    logger.info(f"Created {len(output_paths)} price effect visualizations")
    return output_paths