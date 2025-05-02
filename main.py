#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the Fresh SKU Substitution Analysis pipeline.
"""

import os
import argparse
import pandas as pd
import logging
from datetime import datetime

from src.data.load_data import load_transaction_data, load_product_attributes
from src.data.preprocess import preprocess_transactions
from src.data.create_features import create_pivots, separate_price_changes

from src.analysis.oos_analysis import calculate_oos_substitution, calculate_oos_substitution_with_validation
from src.analysis.price_analysis import calculate_price_effects
from src.analysis.elasticity import calculate_elasticity_matrix
from src.analysis.combined_score import find_top_substitutes, find_substitutes_with_validation

from src.visualization.substitution_network import create_network_visualization
from src.visualization.price_effects import visualize_top_pairs
from src.visualization.report_figures import generate_report_figures

from src.utils.helpers import load_config, ensure_dir, save_results, calculate_timestamp_features, filter_sparse_items, detect_data_anomalies
from src.utils.export_results import export_to_csv

def setup_logging():
    """Set up logging configuration"""
    log_dir = "logs"
    ensure_dir(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"fresh_substitution_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def main(config_path, export_csv=False):
    """
    Main function to run the full pipeline
    
    Parameters:
    -----------
    config_path : str
        Path to configuration file
    export_csv : bool
        Whether to export results to CSV format
    """
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Fresh SKU Substitution Analysis")
    
    # Load configuration
    config = load_config(config_path)
    
    # Create output directories
    ensure_dir(config['data']['interim_dir'])
    ensure_dir(config['data']['results_dir'])
    
    # Load and preprocess data
    logger.info("Loading transaction data")
    transactions_df = load_transaction_data(config['data']['input_file'])
    
    logger.info("Loading product attributes")
    try:
        attributes_df = load_product_attributes(config['data']['attributes_file'])
    except FileNotFoundError:
        logger.warning("Product attributes file not found, continuing without attributes")
        attributes_df = None
    
    logger.info("Preprocessing transaction data")
    transactions_df = preprocess_transactions(transactions_df)
    
    # Check for data anomalies
    logger.info("Checking for data anomalies")
    anomalies = detect_data_anomalies(transactions_df)
    if anomalies:
        logger.warning(f"Detected {len(anomalies)} types of anomalies in the data")
        
    # Filter sparse items if configured
    if config.get('data', {}).get('filter_sparse_items', False):
        min_days = config.get('data', {}).get('min_days', 30)
        logger.info(f"Filtering sparse items (min_days={min_days})")
        transactions_df, filtered_items = filter_sparse_items(
            transactions_df, 
            min_days=min_days
        )
    
    # Add timestamp features for potential use as control variables
    transactions_df = calculate_timestamp_features(transactions_df)
    
    # Create features and pivots
    logger.info("Creating pivot tables")
    sales_pivot, price_pivot, promo_pivot, oos_pivot, control_pivot = create_pivots(
        transactions_df
    )
    
    # Get list of items to analyze
    items_list = sales_pivot.columns.tolist()
    logger.info(f"Analyzing {len(items_list)} unique fresh SKUs")
    
    # Save interim data
    interim_path = os.path.join(config['data']['interim_dir'], 'pivot_tables.pkl')
    logger.info(f"Saving pivot tables to {interim_path}")
    save_results({
        'sales_pivot': sales_pivot,
        'price_pivot': price_pivot,
        'promo_pivot': promo_pivot,
        'oos_pivot': oos_pivot,
        'control_pivot': control_pivot
    }, interim_path)
    
    # Separate price changes into promotion vs price matching
    logger.info("Analyzing price change patterns")
    price_change_types, discount_df = separate_price_changes(
        transactions_df, 
        items_list,
        baseline_window=config['price_analysis']['baseline_window'],
        min_periods=config['price_analysis']['min_baseline_periods'],
        discount_threshold=config['price_analysis']['discount_threshold']
    )
    
    # Create enhanced control variables for regression analyses - always at national level
    # We already have basic control variables in the control_pivot
    # But let's enhance them with more detailed calendar information
    logger.info("Creating enhanced national-level control variables")
    
    # Extract unique dates from the pivot
    unique_dates = sales_pivot.index.get_level_values('date').unique()
    date_df = pd.DataFrame(index=unique_dates)
    
    # Add day of week
    weekday_dummies = pd.get_dummies(pd.DatetimeIndex(unique_dates).weekday, prefix='weekday')
    weekday_dummies.index = unique_dates
    
    # Add month indicators
    month_dummies = pd.get_dummies(pd.DatetimeIndex(unique_dates).month, prefix='month')
    month_dummies.index = unique_dates
    
    # Add quarter indicators
    quarter_dummies = pd.get_dummies(pd.DatetimeIndex(unique_dates).quarter, prefix='quarter')
    quarter_dummies.index = unique_dates
    
    # Add weekend indicator
    is_weekend = pd.DatetimeIndex(unique_dates).weekday.isin([5, 6]).astype(int)
    is_weekend = pd.Series(is_weekend, index=unique_dates, name='is_weekend')
    
    # Add Korean holidays
    try:
        # Define Korean holidays for the analysis period
        korean_holidays = {
            # Traditional Korean holidays (approximation - some are lunar calendar based)
            "new_years_day": [f"{year}-01-01" for year in range(2023, 2026)],
            "seollal": [date for year in range(2023, 2026) 
                       for date in [f"{year}-02-01", f"{year}-02-02", f"{year}-02-03"]],  # Korean New Year
            "independence_day": [f"{year}-03-01" for year in range(2023, 2026)],
            "buddha_birthday": [f"{year}-05-08" for year in range(2023, 2026)],
            "memorial_day": [f"{year}-06-06" for year in range(2023, 2026)],
            "liberation_day": [f"{year}-08-15" for year in range(2023, 2026)],
            "chuseok": [date for year in range(2023, 2026)
                       for date in [f"{year}-09-28", f"{year}-09-29", f"{year}-09-30"]],  # Korean Thanksgiving
            "national_foundation_day": [f"{year}-10-03" for year in range(2023, 2026)],
            "hangul_day": [f"{year}-10-09" for year in range(2023, 2026)],
            "christmas": [f"{year}-12-25" for year in range(2023, 2026)]
        }
        
        # Flatten holiday list
        korean_holiday_dates = []
        for holiday_list in korean_holidays.values():
            korean_holiday_dates.extend(holiday_list)
        
        # Convert to datetime
        korean_holiday_dates = pd.to_datetime(korean_holiday_dates)
        
        # Create holiday indicators
        is_holiday = pd.Series(unique_dates.isin(korean_holiday_dates), index=unique_dates, name='is_holiday')
        
        # Create dummy variables for specific holiday periods
        holiday_dummies = pd.DataFrame(index=unique_dates)
        for holiday_name, dates in korean_holidays.items():
            holiday_dates = pd.to_datetime(dates)
            holiday_dummies[f'holiday_{holiday_name}'] = unique_dates.isin(holiday_dates).astype(int)
        
    except Exception as e:
        logger.warning(f"Error creating Korean holiday calendar: {str(e)}")
        # If holiday calendar fails, create a default series
        is_holiday = pd.Series(False, index=unique_dates, name='is_holiday')
        holiday_dummies = pd.DataFrame(index=unique_dates)
    
    # Combine all calendar controls
    calendar_controls = pd.concat([
        weekday_dummies, 
        month_dummies,
        quarter_dummies,
        is_weekend,
        is_holiday,
        holiday_dummies  # Add specific holiday dummies
    ], axis=1)
    
    # Combine with existing controls
    control_vars = control_pivot.copy()
    for col in calendar_controls.columns:
        control_vars[col] = calendar_controls.reindex(control_vars.index.get_level_values('date'))[col].values

    # Create a parameter to control which analysis methods to use
    use_elasticity = config.get('analysis', {}).get('use_elasticity', True)
    use_validation = config.get('analysis', {}).get('use_validation', True)
    
    # Calculate OOS substitution effects
    logger.info("Calculating OOS substitution effects")
    if use_validation and promo_pivot is not None:
        logger.info("Using enhanced validation for OOS analysis")
        oos_matrix, oos_significance, oos_detailed = calculate_oos_substitution_with_validation(
            sales_pivot, 
            oos_pivot, 
            price_pivot, 
            promo_pivot,
            items_list, 
            min_oos_days=config['analysis']['min_oos_days'],
            control_vars=control_vars
        )
    else:
        oos_matrix, oos_significance = calculate_oos_substitution(
            sales_pivot, 
            oos_pivot, 
            price_pivot, 
            items_list, 
            min_oos_days=config['analysis']['min_oos_days'],
            control_vars=control_vars,
            promo_df=promo_pivot
        )
        oos_detailed = None
    
    # Calculate price effects
    logger.info("Calculating price effects (promotion and price matching)")
    price_matrix, price_type, price_significance = calculate_price_effects(
        sales_pivot, 
        price_pivot, 
        promo_pivot, 
        price_change_types,
        items_list, 
        min_price_changes=config['analysis']['min_price_changes'],
        control_vars=control_vars,
        use_elasticity=use_elasticity,
        oos_df=oos_pivot  # Add OOS data for controlling availability effects
    )
    
    # Calculate elasticity directly if needed
    elasticity_data = None
    if use_elasticity:
        logger.info("Calculating cross-price elasticity between items")
        elasticity_matrix, elasticity_significance = calculate_elasticity_matrix(
            sales_pivot, price_pivot, promo_pivot, items_list, control_vars, oos_pivot
        )
        
        # Store elasticity data for enhanced output
        elasticity_data = {}
        for item_a in items_list:
            elasticity_data[item_a] = {}
            for item_b in items_list:
                if item_a != item_b:
                    elasticity_data[item_a][item_b] = {
                        'elasticity': elasticity_matrix.loc[item_a, item_b],
                        'significant': bool(elasticity_significance.loc[item_a, item_b])
                    }
    
    # Find top substitutes
    logger.info("Finding top substitutes with combined effects")
    if oos_detailed is not None:
        # Use enhanced substitute finding with detailed validation results
        logger.info("Using enhanced substitute finding with validation results")
        substitutes_dict, combined_matrix, _ = find_substitutes_with_validation(
            oos_detailed,
            None,  # No detailed price results yet
            elasticity_data,
            k=config['analysis']['top_k'],
            weights={
                'oos': config['analysis']['weights']['oos'],
                'price': config['analysis']['weights']['price']
            },
            require_significance=config['analysis']['require_significance']
        )
    else:
        # Use standard substitute finding
        substitutes_dict, combined_matrix = find_top_substitutes(
            oos_matrix, 
            oos_significance,
            price_matrix, 
            price_type, 
            price_significance,
            k=config['analysis']['top_k'],
            weights={
                'oos': config['analysis']['weights']['oos'],
                'price': config['analysis']['weights']['price']
            },
            require_significance=config['analysis']['require_significance'],
            elasticity_data=elasticity_data
        )
    
    # Save results
    results_path = os.path.join(config['data']['results_dir'], 'substitution_results.pkl')
    logger.info(f"Saving substitution results to {results_path}")
    
    # Prepare the results dictionary
    results_dict = {
        'substitutes_dict': substitutes_dict,
        'combined_matrix': combined_matrix,
        'oos_matrix': oos_matrix,
        'price_matrix': price_matrix,
        'price_type': price_type,
        'oos_significance': oos_significance,
        'price_significance': price_significance
    }
    
    # Add enhanced results if available
    if 'elasticity_matrix' in locals():
        results_dict['elasticity_matrix'] = elasticity_matrix
        results_dict['elasticity_significance'] = elasticity_significance
        
    if 'oos_detailed' in locals() and oos_detailed is not None:
        results_dict['oos_detailed'] = oos_detailed
        
    if 'elasticity_data' in locals() and elasticity_data is not None:
        results_dict['elasticity_data'] = elasticity_data
    
    # Save the results
    save_results(results_dict, results_path)
    
    # Export to CSV if requested
    if export_csv:
        logger.info("Exporting results to CSV for easier access")
        csv_output_dir = os.path.join(config['data']['results_dir'], 'csv')
        csv_path = export_to_csv(results_path, csv_output_dir)
        logger.info(f"Results exported to CSV: {csv_path}")
    
    # Generate visualizations
    if config['reporting']['generate_visualizations']:
        reports_dir = os.path.join('reports', 'figures')
        ensure_dir(reports_dir)
            
        logger.info("Generating network visualization")
        G, plt = create_network_visualization(
            substitutes_dict, 
            min_score=config['visualization']['network_min_score'],
            max_nodes=config['visualization']['network_max_nodes']
        )
        plt.savefig(os.path.join(reports_dir, 'substitution_network.png'), dpi=300)
        
        logger.info("Generating price effect visualizations for top pairs")
        visualize_top_pairs(
            substitutes_dict,
            transactions_df,
            price_change_types,
            discount_df,
            output_dir=reports_dir
        )
        
        logger.info("Generating report figures")
        generate_report_figures(
            substitutes_dict,
            oos_matrix,
            price_matrix,
            price_type,
            output_dir=reports_dir
        )
    
    logger.info("Analysis pipeline completed successfully")
    
    # Print a summary of results
    summary = f"""
    ===== ANALYSIS SUMMARY =====
    Analyzed {len(items_list)} unique products
    Found substitution relationships for {len([k for k, v in substitutes_dict.items() if v])} products
    Total substitution relationships: {sum(len(v) for v in substitutes_dict.values())}
    
    Results saved to:
    - Pickle format: {results_path}
    """
    
    if export_csv:
        summary += f"- CSV format: {csv_output_dir}\n"
    
    if config['reporting']['generate_visualizations']:
        summary += f"- Visualizations: {reports_dir}\n"
    
    print(summary)
    
    return substitutes_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run Fresh SKU Substitution Analysis')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--csv', action='store_true', help='Export results to CSV format')
    args = parser.parse_args()
    
    main(args.config, export_csv=args.csv)