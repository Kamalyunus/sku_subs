#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network visualization of substitution relationships.
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def create_network_visualization(substitutes_dict, min_score=0.1, max_nodes=50):
    """
    Create a network visualization of substitution relationships
    
    Parameters:
    -----------
    substitutes_dict : dict
        Dictionary of {item: [(substitute_item, score, details), ...]}
    min_score : float
        Minimum score to include in visualization
    max_nodes : int
        Maximum number of nodes to include in visualization
        
    Returns:
    --------
    tuple
        (networkx.Graph, matplotlib.pyplot)
    """
    logger.info(f"Creating substitution network visualization with min_score={min_score}, max_nodes={max_nodes}")
    
    G = nx.DiGraph()
    
    # Add edges for each substitution relationship
    edges = []
    for item_a, substitutes in substitutes_dict.items():
        for sub_info in substitutes:
            item_b = sub_info[0]
            score = sub_info[1]
            details = sub_info[2] if len(sub_info) > 2 else {}
            
            if score >= min_score:
                # Get dominant factor for edge color
                dominant_factor = details.get('dominant_factor', 'unknown')
                edges.append((item_a, item_b, {'weight': score, 'factor': dominant_factor}))
    
    # Sort edges by weight and take top ones if there are too many
    edges.sort(key=lambda x: x[2]['weight'], reverse=True)
    edges = edges[:max_nodes*3]  # Limit total edges
    
    # Add selected edges to graph
    G.add_edges_from(edges)
    
    # Limit to max_nodes by taking highest degree nodes
    if len(G.nodes()) > max_nodes:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        nodes_to_keep = [node for node, degree in top_nodes]
        G = G.subgraph(nodes_to_keep)
    
    # Calculate node sizes based on how often they appear as substitutes
    node_counts = {}
    for item_a, substitutes in substitutes_dict.items():
        if item_a in G.nodes():
            node_counts[item_a] = node_counts.get(item_a, 0) + 1
        
        for sub_info in substitutes:
            item_b = sub_info[0]
            if item_b in G.nodes():
                node_counts[item_b] = node_counts.get(item_b, 0) + 1
    
    # Set node sizes
    node_sizes = [100 + 50 * node_counts.get(node, 0) for node in G.nodes()]
    
    # Set edge colors based on dominant factor
    factor_colors = {
        'availability': 'blue',
        'promotion': 'green',
        'price_matching': 'orange',
        'elasticity': 'purple',
        'unknown': 'gray'
    }
    
    edge_colors = [factor_colors.get(G[u][v]['factor'], 'gray') for u, v in G.edges()]
    
    # Set edge widths based on weights
    edge_weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    
    # Compute layout
    pos = nx.spring_layout(G, seed=42, k=0.15)
    
    # Draw the graph
    plt.figure(figsize=(12, 12))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.6, edge_color=edge_colors, arrows=True)
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title("Product Substitution Network", fontsize=16)
    plt.axis('off')
    
    # Create a legend for edge colors and weights
    for factor, color in factor_colors.items():
        plt.plot([0], [0], linewidth=2, color=color, label=f'{factor.title()}')
    
    for weight in [0.2, 0.5, 0.8]:
        plt.plot([0], [0], linewidth=weight*5, color='gray', linestyle='--',
                 label=f'Strength: {weight:.1f}')
    
    plt.legend(loc='lower right')
    
    logger.info(f"Network visualization created with {len(G.nodes())} nodes and {len(G.edges())} edges")
    return G, plt