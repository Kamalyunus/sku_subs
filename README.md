# Minimized SKU Substitution Analysis

This is a streamlined version of the SKU Substitution Analysis system, focusing on the core functionality of identifying product substitution relationships. The system uses out-of-stock (OOS) patterns, price effects, and promotion impact to find substitute products.

## Core Features

- Identifies substitute/complement products using multiple signals:
  - Out-of-stock substitution effects
  - Price elasticity (substitution or complementary relationships)
  - Promotion cannibalization effects
- Uses statistical validation with control variables 
- Runs two regression models (linear, log-log) to identify relationships
- Exports results in both pickle and CSV formats

## Project Structure

- `data/`: Data storage directory
  - `raw/`: Raw transaction and product data
  - `results/`: Analysis results
  - `generate_sample_data.py`: Script to generate sample data
- `src/`: Source code
  - `data/`: Data loading and preprocessing modules
  - `analysis/`: Core analytical modules
  - `utils/`: Utility functions
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

To export results to CSV format:

```bash
python main.py --config config.yaml --csv
```

To enable detailed logging:

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
- Pickle file with detailed substitution results
- CSV export with primary product, substitute product, and effect scores
- Logs with analysis summary and statistics

This minimized version focuses on computational efficiency and core functionality, removing visualizations and unnecessary processing to deliver faster results with minimal code.