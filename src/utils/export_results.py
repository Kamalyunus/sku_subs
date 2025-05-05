#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Export substitution results from pickle to CSV format.
"""

import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def export_to_csv(results_path, output_dir):
    """
    Export substitution results from pickle file to CSV format
    
    Parameters:
    -----------
    results_path : str
        Path to the pickle file containing results
    output_dir : str
        Directory to save CSV files
    """
    logger.info(f"Exporting results from {results_path} to CSV format")
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Load results
    try:
        results = pd.read_pickle(results_path)
        logger.info("Results loaded successfully")
    except Exception as e:
        logger.error(f"Error loading results: {str(e)}")
        raise
    
    # Extract top substitutes to dataframe
    substitutes_list = []
    for item_a, subs in results['substitutes_dict'].items():
        for item_b, score, details in subs:
            substitutes_list.append({
                'primary_item': item_a,
                'substitute_item': item_b,
                'combined_score': score,
                'oos_effect': details.get('oos_effect', 0),
                'price_effect': details.get('price_effect', 0),
                'promo_effect': details.get('promo_effect', 0),
                'dominant_factor': details.get('dominant_factor', 'unknown'),
                'oos_significant': details.get('oos_significant', False),
                'price_significant': details.get('price_significant', False),
                'promo_significant': details.get('promo_significant', False),
                'price_effect_type': details.get('price_effect_type', 'none'),
                'significant_dimensions': details.get('significant_dimensions', 0)
            })
    
    # Create and save dataframe
    subs_df = pd.DataFrame(substitutes_list)
    subs_csv_path = os.path.join(output_dir, 'top_substitutes.csv')
    subs_df.to_csv(subs_csv_path, index=False)
    logger.info(f"Saved top substitutes to {subs_csv_path}")
    
    # Export matrices to CSV
    for matrix_name in ['combined_matrix', 'oos_matrix', 'price_matrix', 'promo_matrix']:
        if matrix_name in results:
            matrix_path = os.path.join(output_dir, f"{matrix_name}.csv")
            results[matrix_name].to_csv(matrix_path)
            logger.info(f"Saved {matrix_name} to {matrix_path}")
    
    # Export price type matrix if available
    if 'effect_type' in results:
        effect_type_path = os.path.join(output_dir, "effect_type_matrix.csv")
        results['effect_type'].to_csv(effect_type_path)
        logger.info(f"Saved effect type matrix to {effect_type_path}")
        
    # Export significance matrices if available
    for matrix_name in ['oos_significance', 'price_significance', 'promo_significance']:
        if matrix_name in results:
            matrix_path = os.path.join(output_dir, f"{matrix_name}.csv")
            results[matrix_name].to_csv(matrix_path)
            logger.info(f"Saved {matrix_name} to {matrix_path}")
    
    logger.info("Export to CSV complete")
    return subs_csv_path

if __name__ == "__main__":
    import argparse
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )
    
    parser = argparse.ArgumentParser(description="Export substitution results to CSV")
    parser.add_argument("--input", default="data/results/substitution_results.pkl", 
                        help="Path to input pickle file")
    parser.add_argument("--output-dir", default="data/results/csv", 
                        help="Directory to save CSV files")
    
    args = parser.parse_args()
    
    try:
        output_path = export_to_csv(args.input, args.output_dir)
        print(f"Results exported successfully to {args.output_dir}")
        print(f"Primary substitution CSV file: {output_path}")
        sys.exit(0)
    except Exception as e:
        print(f"Error exporting results: {str(e)}")
        sys.exit(1)