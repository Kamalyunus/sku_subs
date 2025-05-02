#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate sample data for SKU substitution analysis.
This script creates realistic transaction data with patterns
that demonstrate substitution effects.
"""

import pandas as pd
import numpy as np
import os
import random
from datetime import datetime, timedelta

def generate_sample_data(
    output_dir="raw",
    start_date="2024-01-01",
    days=365,
    products=100,
    out_of_stock_events=True,
    price_changes=True,
    promotions=True,
    random_seed=42
):
    """
    Generate sample transaction data for substitution analysis
    
    Parameters:
    -----------
    output_dir : str
        Directory to save data (within data/)
    start_date : str
        Start date for transaction data
    days : int
        Number of days to generate
    products : int
        Number of products to include
    out_of_stock_events : bool
        Whether to include out-of-stock events
    price_changes : bool
        Whether to include price changes
    promotions : bool
        Whether to include promotions
    random_seed : int
        Random seed for reproducibility
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(os.path.join("data", output_dir)):
        os.makedirs(os.path.join("data", output_dir))
    
    # Generate product data
    print("Generating product data...")
    
    # Create product categories with standardized SKU codes
    product_categories = {
        "Fruit": [],
        "Vegetables": [],
        "Dairy": []
    }
    
    # Generate SKUs for each category
    sku_counter = 1
    for category in product_categories:
        # Allocate roughly equal number of products to each category
        category_count = products // len(product_categories)
        if category == list(product_categories.keys())[0]:
            category_count += products % len(product_categories)  # Add remainder to first category
            
        # Create SKUs for this category
        for _ in range(category_count):
            sku_id = f"SKU_{sku_counter:03d}"
            product_categories[category].append(sku_id)
            sku_counter += 1
    
    # Create list of all products
    all_products = []
    for category, products_list in product_categories.items():
        for sku in products_list:
            all_products.append((sku, category))
    
    # Select subset of products if needed
    if products < len(all_products):
        selected_products = random.sample(all_products, products)
    else:
        selected_products = all_products[:products]  # Ensure we don't exceed the requested number
    
    # Generate base prices for products
    product_data = {}
    for product, category in selected_products:
        if category == "Fruit":
            base_price = round(random.uniform(0.5, 4.0), 2)
        elif category == "Vegetables":
            base_price = round(random.uniform(0.8, 3.5), 2)
        else:  # Dairy
            base_price = round(random.uniform(1.5, 5.0), 2)
        
        # Generate baseline sales volume
        base_sales = int(random.uniform(10, 70))
        
        product_data[product] = {
            "category": category,
            "base_price": base_price,
            "base_sales": base_sales
        }
    
    # Create substitution relationships
    print("Creating substitution relationships...")
    
    # Group products by category
    products_by_category = {}
    for product, category in selected_products:
        if category not in products_by_category:
            products_by_category[category] = []
        products_by_category[category].append(product)
    
    # Define substitution relationships
    substitution_effects = {}
    
    for category, category_products in products_by_category.items():
        for i, product_a in enumerate(category_products):
            substitution_effects[product_a] = {}
            
            # Create substitution effects with other products in same category
            for j, product_b in enumerate(category_products):
                if product_a != product_b:
                    # Create substitution clusters based on SKU numbers
                    # Products with close SKU numbers have stronger substitution effects
                    # Extract numeric part of SKU IDs
                    sku_a_num = int(product_a.split('_')[1])
                    sku_b_num = int(product_b.split('_')[1])
                    
                    # Calculate numeric distance (close numbers = similar products)
                    distance = abs(sku_a_num - sku_b_num)
                    
                    if distance <= 5:  # Similar products (close SKU numbers)
                        effect = random.uniform(0.4, 0.7)  # Strong effect
                    else:
                        effect = random.uniform(0.1, 0.3)  # Weaker effect
                    
                    substitution_effects[product_a][product_b] = effect
    
    # Generate transaction data
    print("Generating transaction data...")
    
    # Create date range
    start = datetime.strptime(start_date, "%Y-%m-%d")
    date_range = [start + timedelta(days=i) for i in range(days)]
    
    # Generate transactions
    transactions = []
    
    # Track out-of-stock events and promotions
    oos_events = {}
    promotions_schedule = {}
    price_change_events = {}
    
    # Randomly schedule out-of-stock events, promotions, and price changes
    if out_of_stock_events:
        for product in product_data:
            # Random OOS events (10% chance for each product)
            if random.random() < 0.3:
                oos_start = random.randint(5, days - 5)
                oos_duration = random.randint(2, 5)
                oos_events[product] = (oos_start, oos_start + oos_duration)
    
    if promotions:
        for product in product_data:
            # Random promotions (20% chance for each product)
            if random.random() < 0.3:
                promo_start = random.randint(3, days - 7)
                promo_duration = random.randint(3, 7)
                discount = round(random.uniform(0.1, 0.3), 2)  # 10-30% discount
                promotions_schedule[product] = (promo_start, promo_start + promo_duration, discount)
    
    if price_changes:
        for product in product_data:
            # Random price changes (15% chance for each product)
            if random.random() < 0.3:
                change_day = random.randint(10, days - 3)
                # Price can go up or down
                change_pct = round(random.uniform(-0.15, 0.15), 2)
                price_change_events[product] = (change_day, change_pct)
    
    # Generate daily transactions
    for day_idx, date in enumerate(date_range):
        for product in product_data:
            # Check if product is out of stock
            is_oos = False
            if product in oos_events:
                oos_start, oos_end = oos_events[product]
                if oos_start <= day_idx < oos_end:
                    is_oos = True
            
            # Get base price and adjust for promotions and price changes
            price = product_data[product]["base_price"]
            is_promotion = False
            
            # Apply promotion if scheduled
            if product in promotions_schedule:
                promo_start, promo_end, discount = promotions_schedule[product]
                if promo_start <= day_idx < promo_end:
                    price = round(price * (1 - discount), 2)
                    is_promotion = True
            
            # Apply price change if scheduled
            if product in price_change_events and not is_promotion:
                change_day, change_pct = price_change_events[product]
                if day_idx >= change_day:
                    price = round(price * (1 + change_pct), 2)
            
            # Calculate sales with randomness and seasonal patterns
            base_sales = product_data[product]["base_sales"]
            
            # Add weekday effect (higher on weekends)
            weekday = date.weekday()
            weekday_factor = 1.2 if weekday >= 5 else 1.0
            
            # Add random noise
            noise_factor = random.uniform(0.8, 1.2)
            
            # Calculate sales before substitution effects
            sales = base_sales * weekday_factor * noise_factor
            
            # Apply substitution effects
            if out_of_stock_events:
                for other_product, effect in substitution_effects.get(product, {}).items():
                    if other_product in oos_events:
                        other_oos_start, other_oos_end = oos_events[other_product]
                        if other_oos_start <= day_idx < other_oos_end:
                            # Increase sales when substitute is out of stock
                            sales += base_sales * effect * random.uniform(0.8, 1.2)
            
            # Apply price effects
            if promotions or price_changes:
                for other_product, effect in substitution_effects.get(product, {}).items():
                    # Check for promotions
                    if other_product in promotions_schedule:
                        other_promo_start, other_promo_end, other_discount = promotions_schedule[other_product]
                        if other_promo_start <= day_idx < other_promo_end:
                            # Decrease sales when substitute is on promotion
                            sales -= base_sales * effect * other_discount * random.uniform(0.5, 1.0)
            
            # Reduce sales if out of stock (not always zero)
            if is_oos:
                # Sometimes zero, sometimes highly reduced, sometimes moderately reduced
                oos_effect_type = random.randint(0, 10)
                if oos_effect_type < 5:  # 50% chance of zero sales
                    sales = 0
                elif oos_effect_type < 8:  # 30% chance of severely reduced (5-20% of normal)
                    sales = sales * random.uniform(0.05, 0.2)
                else:  # 20% chance of moderately reduced (20-50% of normal)
                    sales = sales * random.uniform(0.2, 0.5)
            
            # Ensure sales is non-negative and round to integer
            sales = max(0, int(round(sales)))
            
            # Add transaction record
            transactions.append({
                "date": date.strftime("%Y-%m-%d"),
                "item_id": product,
                "sales": sales,
                "price": price,
                "is_on_promotion": int(is_promotion),
                "is_out_of_stock": int(is_oos)
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(transactions)
    
    # Create product attributes data with additional metadata
    product_attributes = []
    
    # Create some product names and subcategories to make data more realistic
    fruit_names = ["Apple", "Banana", "Orange", "Pear", "Grapes", "Strawberry", "Blueberry", 
                   "Kiwi", "Mango", "Pineapple", "Watermelon", "Peach", "Cherry", "Lemon"]
    vegetable_names = ["Carrot", "Broccoli", "Lettuce", "Cucumber", "Tomato", "Potato", "Onion",
                       "Pepper", "Zucchini", "Spinach", "Kale", "Cabbage", "Mushroom", "Celery"]
    dairy_names = ["Milk", "Yogurt", "Cheese", "Butter", "Cream", "Cottage Cheese", "Sour Cream",
                  "Ice Cream", "Whipped Cream", "Ghee", "Kefir", "Custard"]
    
    subcategory_map = {
        "Fruit": ["Fresh", "Organic", "Frozen", "Dried", "Canned", "Imported"],
        "Vegetables": ["Fresh", "Organic", "Frozen", "Canned", "Local", "Imported"],
        "Dairy": ["Regular", "Organic", "Low-fat", "Whole", "Plant-based", "Lactose-free"]
    }
    
    for product, data in product_data.items():
        category = data["category"]
        sku_num = int(product.split('_')[1])
        
        # Assign a product name based on category and SKU number
        if category == "Fruit":
            names_list = fruit_names
        elif category == "Vegetables":
            names_list = vegetable_names
        else:  # Dairy
            names_list = dairy_names
            
        product_name = names_list[sku_num % len(names_list)]
        subcategory = subcategory_map[category][sku_num % len(subcategory_map[category])]
        
        product_attributes.append({
            "item_id": product,
            "product_name": product_name,
            "category": category,
            "sub_category": subcategory
        })
    
    attributes_df = pd.DataFrame(product_attributes)
    
    # Save files
    transactions_file = os.path.join("data", output_dir, "fresh_transactions.csv")
    attributes_file = os.path.join("data", output_dir, "product_attributes.csv")
    
    df.to_csv(transactions_file, index=False)
    attributes_df.to_csv(attributes_file, index=False)
    
    print(f"Generated {len(df)} transaction records for {len(product_data)} products")
    print(f"Transaction data saved to: {transactions_file}")
    print(f"Product attributes saved to: {attributes_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate sample data for SKU substitution analysis")
    parser.add_argument("--output-dir", default="raw", help="Output directory (within data/)")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--days", type=int, default=365, help="Number of days to generate")
    parser.add_argument("--products", type=int, default=100, help="Number of products")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    generate_sample_data(
        output_dir=args.output_dir,
        start_date=args.start_date,
        days=args.days,
        products=args.products,
        random_seed=args.seed
    )