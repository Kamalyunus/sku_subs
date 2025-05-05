# Interpreting Substitution Analysis Results

This guide explains how to interpret the CSV files produced by the substitution analysis pipeline.

## CSV Output Files

The analysis generates several CSV files in the `data/results/csv/` directory:

| File Name | Description |
|-----------|-------------|
| **top_substitutes.csv** | Primary output with ranked substitute items for each product |
| **combined_matrix.csv** | Full matrix of combined scores between all product pairs |
| **oos_matrix.csv** | Matrix of out-of-stock (OOS) effects between products |
| **price_matrix.csv** | Matrix of price elasticity effects between products |
| **promo_matrix.csv** | Matrix of promotion effects between products |
| **oos_significance.csv** | Boolean matrix indicating statistical significance of OOS effects |
| **price_significance.csv** | Boolean matrix indicating statistical significance of price effects |
| **promo_significance.csv** | Boolean matrix indicating statistical significance of promo effects |
| **effect_type_matrix.csv** | Categorization of relationships as "substitute", "complement", or "none" |

## Understanding top_substitutes.csv

This is the main output file, containing the top substitutes for each product ranked by combined score.

### Columns in top_substitutes.csv

| Column | Description | Interpretation |
|--------|-------------|----------------|
| **primary_item** | ID of the primary product | The product for which substitutes are being identified |
| **substitute_item** | ID of the potential substitute | A product that customers may choose as an alternative |
| **combined_score** | Overall substitution score (0-1) | Higher values indicate stronger substitution relationship |
| **oos_effect** | Out-of-stock effect | Relative sales increase of substitute when primary is OOS |
| **price_effect** | Price elasticity | % change in substitute sales per 1% change in primary price |
| **promo_effect** | Promotion effect | Sales impact on substitute when primary is on promotion |
| **dominant_factor** | Main driver of the relationship | "availability", "elasticity", or "promotion" |
| **oos_significant** | Statistical significance of OOS effect | True if the OOS effect is statistically significant |
| **price_significant** | Statistical significance of price effect | True if the price effect is statistically significant |
| **promo_significant** | Statistical significance of promo effect | True if the promo effect is statistically significant |
| **price_effect_type** | Type of price relationship | Typically "elasticity" for the calculation method used |
| **significant_dimensions** | Count of significant effects | Number of effects (0-3) that are statistically significant |

### Interpreting Effect Values

#### 1. OOS Effect (Out of Stock Effect)

The OOS effect shows how much the substitute item's sales increase when the primary item is out of stock.

| Value Range | Interpretation |
|-------------|----------------|
| **0.0** | No substitution when primary is out of stock |
| **0.1-0.3** | Weak substitution (10-30% sales lift) |
| **0.3-0.6** | Moderate substitution (30-60% sales lift) |
| **0.6-1.0** | Strong substitution (60-100% sales lift) |
| **>1.0** | Very strong substitution (sales more than double) |

#### 2. Price Effect (Cross-Price Elasticity)

The price effect represents the cross-price elasticity between products.

| Value Range | Interpretation |
|-------------|----------------|
| **>0** | **Substitute Relationship**: Sales of substitute increase when primary price increases |
| **<0** | **Complementary Relationship**: Sales of substitute decrease when primary price increases |
| **0.1-0.5** | Low elasticity |
| **0.5-1.0** | Moderate elasticity |
| **1.0-2.0** | High elasticity |
| **>2.0** | Very high elasticity |

#### 3. Promo Effect

The promo effect shows how much the substitute's sales decrease when the primary item is on promotion.

| Value Range | Interpretation |
|-------------|----------------|
| **0.0** | No promotional impact |
| **0.1-0.3** | Weak cannibalization (10-30% sales reduction) |
| **0.3-0.6** | Moderate cannibalization |
| **0.6-1.0** | Strong cannibalization |
| **>1.0** | Extreme cannibalization |

### Interpreting the Combined Score

The combined score represents the overall strength of the substitution relationship.

| Score Range | Interpretation | Recommended Action |
|-------------|----------------|-------------------|
| **0.8-1.0** | Strong substitutes | Consider as direct replacements, ensure at least one is in stock |
| **0.5-0.8** | Moderate substitutes | Important alternatives, monitor jointly |
| **0.3-0.5** | Weak substitutes | Secondary alternatives |
| **0.1-0.3** | Very weak substitutes | Limited substitutability |
| **<0.1** | Minimal substitution | Not practical substitutes |

## Using Matrix Files

The matrix files provide the full relationship data between all product pairs.

### Reading Matrix Files

Each matrix is indexed by product IDs in both rows and columns:
- Row product is the primary item (A)
- Column product is the substitute item (B)
- Value in cell [A,B] shows the effect of A on B

For example, in `price_matrix.csv`, a value of 0.5 at row "apple" and column "orange" means a 1% increase in apple price leads to a 0.5% increase in orange sales.

### Effect Type Matrix

The `effect_type_matrix.csv` categorizes each product pair relationship:

| Value | Interpretation |
|-------|----------------|
| **substitute** | Products are substitutes (positive price elasticity) |
| **complement** | Products are complements (negative price elasticity) |
| **none** | No clear relationship detected |

## Business Applications

### Inventory Management
- Ensure at least one product from each strong substitute group is in stock
- Adjust safety stock levels based on substitution patterns

### Pricing Strategy
- Consider price elasticity when planning price changes
- Coordinate pricing among substitute products
- Be cautious about raising prices on both complementary products

### Merchandising
- Place substitutes near each other for out-of-stock situations
- Consider joint displays or promotions for complementary products

### Promotion Planning
- Avoid promoting strong substitutes simultaneously
- Consider promoting complementary products together

## Tips for Analysis

1. **Focus on significant effects**: Pay attention to the significance columns
2. **Examine the dominant factor**: Understand what drives each relationship
3. **Consider effect magnitude**: Larger effects indicate stronger relationships
4. **Look for complementary pairs**: Negative price effects identify complementary products
5. **Use multiple dimensions**: The most reliable substitutes have multiple significant effects