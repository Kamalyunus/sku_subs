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
  - `generate_sample_data.py`: Script to generate sample data
- `src/`: Source code
  - `substitution_analysis.py`: Consolidated analysis module
- `logs/`: Log files
- `main.py`: Main pipeline script
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

## Architecture

The code is highly optimized with minimal dependencies:
- Single consolidated module (`substitution_analysis.py`) 
- Statistical significance filtering (p < 0.05)
- Only statistically valid effects are included
- Streamlined processing without visualization overhead

This architecture focuses on delivering accurate substitution relationships through a simplified codebase with just the essential functionality needed.