# Minimized SKU Substitution Analysis

This is a highly optimized version of the SKU Substitution Analysis system, focusing on the core functionality of identifying product substitution relationships. The system uses out-of-stock (OOS) patterns, price effects, and promotion impact to find substitute products with minimal code.

## Core Features

- Identifies substitute/complement products using multiple signals:
  - Out-of-stock substitution effects
  - Price elasticity (substitution or complementary relationships)
  - Promotion cannibalization effects
- Uses statistical validation with control variables 
- Runs two regression models (linear, log-log) to identify relationships
- Exports results directly to CSV format

## Project Structure

- `data/`: Data storage directory
  - `raw/`: Raw transaction and product data
  - `results/`: Analysis results
    - `figures/`: Visualization output (generated when using visualize_sku_pair.py)
  - `generate_sample_data.py`: Script to generate sample data
- `src/`: Source code
  - `substitution_analysis.py`: Consolidated analysis module
  - `visualize_relationship.py`: Visualization functions
- `logs/`: Log files
- `main.py`: Main pipeline script
- `visualize_sku_pair.py`: SKU pair visualization tool
- `config.yaml`: Configuration parameters

## Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Generating Sample Data

Generate sample transaction data with:

```bash
python data/generate_sample_data.py
```

### Running Analysis

Run the analysis pipeline with:

```bash
python main.py --config config.yaml
```

All analysis results are automatically exported to CSV format. To enable detailed logging:

```bash
python main.py --config config.yaml --verbose
```

## Data Format

The system requires transaction data with the following columns:
- `date`: Date of transaction (YYYY-MM-DD)
- `item_id`: Unique product identifier
- `sales`: Quantity sold
- `price`: Product price
- `is_on_promotion`: Boolean flag (0/1) for promotions
- `is_out_of_stock`: Boolean flag (0/1) for OOS status

The product attributes file (optional) should include:
- `item_id`: Unique product identifier
- `category`: Main product category
- `sub_category`: Sub-category

## Configuration

The `config.yaml` file contains essential parameters for the analysis:

- Data paths and preprocessing settings
- Analysis thresholds (min OOS days, significance requirements)
- Substitution scope (category, sub_category, or all)
- Minimum data requirements (days per product)

### Substitution Scope

Control the scope of substitution relationships:

```yaml
analysis:
  substitution_scope: "category"  # Options: "category", "sub_category", "all"
```

## Results

The system outputs:
- CSV export (`substitutes.csv`) with the following columns:
  - `primary_item`: The main product ID for which substitutes are being identified
  - `substitute_item`: The potential substitute product ID 
  - `combined_score`: Normalized overall substitution score (higher values indicate stronger substitution relationships)
  - `oos_effect`: Relative increase in substitute item sales when the primary item is out of stock (range 0-5, where 0 means no effect and higher values indicate stronger substitution)
  - `price_effect`: Cross-price elasticity coefficient from log-log model (positive values indicate substitutes; higher values mean stronger price sensitivity)
  - `promo_effect`: Impact of primary item promotions on substitute item sales (negative values indicate cannibalization)
  - `relationship_type`: Categorization of the relationship:
      - "Substitute": When OOS or price effects are positive and statistically significant (p<0.05)
      - "Complement": When price effects are negative or promotion effects are negative, and statistically significant
      - "Undefined": When no effects meet the statistical significance threshold

Only statistically significant effects (p<0.05) are included in the calculations, and all effects are capped at reasonable maximum values to prevent unrealistic outliers.

- Logs with analysis summary and statistics

## Visualization

The system includes a minimal visualization tool for analyzing specific SKU pairs:

```bash
python visualize_sku_pair.py --item-a ITEM001 --item-b ITEM002
```

This creates three plots showing the relationship between the two products:
1. **OOS Relationship**: Shows sales of item A with out-of-stock periods of item B highlighted
2. **Price Relationship**: Shows sales of item A compared to price changes of item B
3. **Promotion Relationship**: Shows sales of item A with promotion periods of item B highlighted

Visualizations are saved to `data/results/figures/` by default.

Optional arguments:
- `--config CONFIG_PATH`: Specify an alternative config file
- `--output-dir OUTPUT_DIR`: Specify where to save visualizations
- `--verbose`: Enable detailed logging

The visualization component is optional and requires matplotlib (`pip install matplotlib`).

## Architecture

The code is highly optimized with minimal dependencies:
- Single consolidated module (`substitution_analysis.py`) 
- Statistical significance filtering (p < 0.05)
- Only statistically valid effects are included
- Streamlined processing with optional visualization tools

This architecture focuses on delivering accurate substitution relationships through a simplified codebase with essential functionality and visualization only when needed.