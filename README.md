# Fresh SKU Substitution Analysis

This project provides a comprehensive system for identifying product substitution relationships in fresh SKUs based on transaction data. It uses out-of-stock (OOS) substitution patterns, price effects (promotions and competitive price matching), and cross-price elasticity to find the most relevant substitutes for each product.

## Features

- Identifies top substitutes for each fresh SKU using multiple methodologies
- Analyzes substitution through out-of-stock patterns with advanced statistical validation
- Distinguishes between promotions and competitive price matching
- Calculates cross-price elasticity between products for formal economic analysis
- Provides statistical validation of substitution relationships with control variables
- Creates visualizations for substitution networks and price effects
- Handles large-scale retail data efficiently with data quality checks
- Exports results in both pickle and CSV formats for easy access

## Enhanced Analytics

The system includes these advanced analytical capabilities:

- **Cross-Price Elasticity**: Formal economic measurement of substitution effects
- **Enhanced Statistical Validation**: Control for time effects and other variables
- **Data Quality Monitoring**: Detection of anomalies and sparse data patterns
- **Combined Analytical Framework**: Integration of multiple substitution signals

## Project Structure

- `data/`: Data storage directory
  - `raw/`: Raw transaction and product data
  - `interim/`: Intermediate data files
  - `processed/`: Processed data ready for analysis
  - `results/`: Analysis results and outputs
  - `generate_sample_data.py`: Script to generate sample transaction data
- `src/`: Source code
  - `data/`: Data loading and preprocessing modules
  - `analysis/`: Core analytical modules including elasticity and validation
  - `visualization/`: Data visualization tools
  - `utils/`: Utility functions for data handling and export
- `reports/`: Generated reports and visualizations
- `logs/`: Log files from analysis runs
- `main.py`: Unified main script for the entire pipeline
- `config.yaml`: Configuration parameters for the analysis

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
python data/generate_sample_data.py --products 100 --days 365
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

## Data Format

The system expects transaction data with the following columns:
- `date`: Date of transaction (YYYY-MM-DD)
- `item_id`: Unique product identifier
- `sales`: Quantity sold
- `price`: Product price
- `is_on_promotion`: Boolean flag (0/1) indicating if product was on promotion
- `is_out_of_stock`: Boolean flag (0/1) indicating if product was out of stock

## Configuration

The `config.yaml` file contains all parameters for the analysis pipeline, including:

- Data paths and preprocessing parameters
- Analysis thresholds and weights (OOS vs price effects)
- Minimum data requirements (days per product, price changes)
- Visualization settings and network parameters
- Reporting options and export formats
- Elasticity calculation and validation parameters

## Results

The system generates:

- Comprehensive substitution matrices and detailed relationships
- Network visualization of top substitutes
- Price effect plots for top substitution pairs
- Various analytical charts showing substitution patterns
- CSV exports for easy data access

## Visualization Outputs

The analysis produces several types of visualizations:

1. **Substitution Network**: Shows products and their substitution relationships
2. **Price Effect Plots**: Visualizes how price changes affect substitution patterns
3. **Analysis Charts**: 
   - Substitution score distributions
   - Out-of-stock effect heatmaps
   - Dominant factor breakdown (OOS vs price effects)
   - Number of substitutes per product