import unittest
import os
import tempfile
import pandas as pd
import numpy as np
import logging
from unittest.mock import patch, mock_open

# Add src to Python path to allow direct import of substitution_analysis
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) # To import generate_sample_data

from substitution_analysis import load_sku_embeddings, calculate_embedding_similarity, find_substitutes, export_to_csv

# Suppress logging during tests to keep output clean, can be enabled for debugging
logging.disable(logging.CRITICAL)

class TestLoadSkuEmbeddings(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory to store CSV files
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        # Clean up the temporary directory
        self.temp_dir.cleanup()

    def test_load_valid_embeddings(self):
        file_content = "item_id,embedding\nSKU1,0.1,0.2,0.3\nSKU2,0.4,0.5,0.6"
        file_path = os.path.join(self.temp_dir.name, "valid_embeddings.csv")
        with open(file_path, "w") as f:
            f.write(file_content)

        expected = {
            "SKU1": [0.1, 0.2, 0.3],
            "SKU2": [0.4, 0.5, 0.6],
        }
        result = load_sku_embeddings(file_path)
        self.assertEqual(result, expected)

    def test_load_embeddings_file_not_found(self):
        with patch('logging.Logger.error') as mock_log_error:
            result = load_sku_embeddings("non_existent_file.csv")
            self.assertEqual(result, {})
            mock_log_error.assert_called_with("SKU embeddings file not found: non_existent_file.csv")

    def test_load_embeddings_malformed_csv_missing_column(self):
        file_content = "item_id,wrong_column_name\nSKU1,0.1,0.2,0.3"
        file_path = os.path.join(self.temp_dir.name, "malformed_missing_col.csv")
        with open(file_path, "w") as f:
            f.write(file_content)
        
        with patch('logging.Logger.error') as mock_log_error:
            result = load_sku_embeddings(file_path)
            self.assertEqual(result, {})
            mock_log_error.assert_called_with("SKU embeddings file must contain 'item_id' and 'embedding' columns")

    def test_load_embeddings_malformed_csv_bad_value(self):
        file_content = "item_id,embedding\nSKU1,0.1,not_a_float,0.3"
        file_path = os.path.join(self.temp_dir.name, "malformed_bad_val.csv")
        with open(file_path, "w") as f:
            f.write(file_content)

        with patch('logging.Logger.error') as mock_log_error:
            result = load_sku_embeddings(file_path)
            self.assertEqual(result, {"SKU1": []}) # Should skip the bad item, or return empty for SKU1
            # More precise check of the log:
            found_log = False
            for call_args in mock_log_error.call_args_list:
                if "Error parsing embedding for item_id SKU1" in call_args[0][0]:
                    found_log = True
                    break
            self.assertTrue(found_log, "Expected log message for parsing error not found.")
            # The current implementation returns an empty dict if any error occurs during parsing an item.
            # Let's adjust the expectation based on the actual behavior from the previous subtask:
            # If a specific row fails, it's skipped.
            self.assertEqual(load_sku_embeddings(file_path), {})


    def test_load_embeddings_empty_csv(self):
        file_content = "item_id,embedding\n" # Header only
        file_path = os.path.join(self.temp_dir.name, "empty.csv")
        with open(file_path, "w") as f:
            f.write(file_content)
        result = load_sku_embeddings(file_path)
        self.assertEqual(result, {})

    def test_load_embeddings_completely_empty_file(self):
        file_path = os.path.join(self.temp_dir.name, "completely_empty.csv")
        with open(file_path, "w") as f:
            f.write("") # No content at all
        result = load_sku_embeddings(file_path)
        self.assertEqual(result, {})


class TestCalculateEmbeddingSimilarity(unittest.TestCase):
    def test_similarity_identical_vectors(self):
        v1 = [1, 2, 3]
        v2 = [1, 2, 3]
        self.assertAlmostEqual(calculate_embedding_similarity(v1, v2), 1.0)

    def test_similarity_orthogonal_vectors(self):
        v1 = [1, 0, 0]
        v2 = [0, 1, 0]
        self.assertAlmostEqual(calculate_embedding_similarity(v1, v2), 0.0)

    def test_similarity_opposite_vectors(self):
        v1 = [1, 2, 3]
        v2 = [-1, -2, -3]
        self.assertAlmostEqual(calculate_embedding_similarity(v1, v2), -1.0)

    def test_similarity_known_values(self):
        v1 = [1, 2] # norm sqrt(5)
        v2 = [2, 3] # norm sqrt(13)
        # dot_product = 1*2 + 2*3 = 2 + 6 = 8
        # similarity = 8 / (sqrt(5) * sqrt(13)) = 8 / sqrt(65)
        expected_similarity = 8 / np.sqrt(65)
        self.assertAlmostEqual(calculate_embedding_similarity(v1, v2), expected_similarity)

    def test_similarity_zero_vector(self):
        v1 = [0, 0, 0]
        v2 = [1, 2, 3]
        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v1, v2), 0.0)
            mock_log_warn.assert_called_with("calculate_embedding_similarity: One or both embeddings have zero norm.")
        
        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v2, v1), 0.0) # Order shouldn't matter
            mock_log_warn.assert_called_with("calculate_embedding_similarity: One or both embeddings have zero norm.")

        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v1, v1), 0.0) # Both zero
            mock_log_warn.assert_called_with("calculate_embedding_similarity: One or both embeddings have zero norm.")


    def test_similarity_mismatched_dimensions(self):
        v1 = [1, 2, 3]
        v2 = [1, 2]
        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v1, v2), 0.0)
            mock_log_warn.assert_called_with(f"calculate_embedding_similarity: Embedding dimensions do not match: {np.array(v1).shape} vs {np.array(v2).shape}")

    def test_similarity_empty_input(self):
        v1 = []
        v2 = [1,2]
        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v1,v2), 0.0)
            mock_log_warn.assert_called_with("calculate_embedding_similarity: One or both embeddings are empty.")
        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v2,v1), 0.0)
            mock_log_warn.assert_called_with("calculate_embedding_similarity: One or both embeddings are empty.")
        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v1,v1), 0.0)
            mock_log_warn.assert_called_with("calculate_embedding_similarity: One or both embeddings are empty.")

    def test_similarity_invalid_input_none(self):
        v1 = None
        v2 = [1,2]
        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v1,v2), 0.0)
            mock_log_warn.assert_called_with("calculate_embedding_similarity: One or both embeddings are None.")
        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v2,v1), 0.0)
            mock_log_warn.assert_called_with("calculate_embedding_similarity: One or both embeddings are None.")
        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v1,v1), 0.0)
            mock_log_warn.assert_called_with("calculate_embedding_similarity: One or both embeddings are None.")

    def test_similarity_invalid_input_type(self):
        v1 = "not a list"
        v2 = [1,2]
        with patch('logging.Logger.warning') as mock_log_warn:
            self.assertAlmostEqual(calculate_embedding_similarity(v1,v2), 0.0)
            mock_log_warn.assert_called_with("calculate_embedding_similarity: Embeddings must be lists or numpy arrays.")


class TestFindSubstitutes(unittest.TestCase):
    def setUp(self):
        # Sample detailed_results for testing
        self.detailed_results = {
            "SKU1": {
                "SKU2": { # Strong stat, good substitute
                    'validation_successful': True, 'oos_effect': 2.0, 'oos_significant': True,
                    'price_effect': 1.0, 'price_significant': True, 'promo_effect': 0.0, 'promo_significant': False,
                    'relationship_type': 'Substitute'
                },
                "SKU3": { # Weak stat, potential complement via promo
                    'validation_successful': True, 'oos_effect': 0.1, 'oos_significant': False,
                    'price_effect': -0.5, 'price_significant': True, 'promo_effect': -0.5, 'promo_significant': True,
                    'relationship_type': 'Complement'
                },
                "SKU4": { # No significant stat effects
                    'validation_successful': True, 'oos_effect': 0.1, 'oos_significant': False,
                    'price_effect': 0.05, 'price_significant': False, 'promo_effect': 0.0, 'promo_significant': False,
                     'relationship_type': 'Undefined'
                }
            },
            "SKU2": {
                "SKU1": { # Strong stat, good substitute for SKU1
                    'validation_successful': True, 'oos_effect': 1.5, 'oos_significant': True,
                    'price_effect': 0.8, 'price_significant': True, 'promo_effect': 0.0, 'promo_significant': False,
                    'relationship_type': 'Substitute'
                }
            },
            "SKU3": {}, # No substitutes found or analyzed for SKU3
            "SKU4": { # SKU4 has a substitute SKU5
                 "SKU5": {
                    'validation_successful': True, 'oos_effect': 3.0, 'oos_significant': True,
                    'price_effect': 1.0, 'price_significant': True, 'promo_effect': 0.0, 'promo_significant': False,
                    'relationship_type': 'Substitute'
                 }
            },
            "SKU5": {}
        }
        # Sample SKU embeddings
        self.sku_embeddings = {
            "SKU1": [1.0, 0.0, 0.0], # Orthogonal to SKU3
            "SKU2": [0.9, 0.1, 0.0], # Very similar to SKU1
            "SKU3": [0.0, 1.0, 0.0], # Orthogonal to SKU1
            "SKU4": [0.5, 0.5, 0.5], # SKU4 embedding available
            # SKU5 embedding is missing
        }
        # Max stat score based on constants in substitution_analysis.py
        # MAX_OOS_EFFECT = 5.0, MAX_PRICE_EFFECT = 3.0, MAX_PROMO_EFFECT = 3.0
        self.max_stat_score = 5.0 + 3.0 + 3.0 # = 11.0

    def test_find_substitutes_no_embeddings_data(self):
        # Test with empty sku_embeddings, weight doesn't matter if no embeddings
        subs = find_substitutes(self.detailed_results, {}, embedding_weight=0.5, k=2)
        
        # SKU1 -> SKU2 (stat score: 2.0 (oos) + 1.0 (price) = 3.0) / 11.0 = 0.2727
        # SKU1 -> SKU3 (stat score: 0) because complement and oos_effect is not > 0
        # SKU1 -> SKU4 (stat score: 0)
        self.assertTrue("SKU2" in [s[0] for s in subs.get("SKU1", [])])
        sku1_sku2_details = next(s[2] for s in subs.get("SKU1", []) if s[0] == "SKU2")
        self.assertAlmostEqual(sku1_sku2_details['embedding_similarity_score'], 0.0) # No embedding data
        self.assertAlmostEqual(sku1_sku2_details['scaled_embedding_similarity_score'], 0.0) # No embedding data
        
        # Check score for SKU1 -> SKU2
        # Stat contribution: oos_effect (2.0) + price_effect (1.0) = 3.0
        # Normalized stat score: 3.0 / 11.0 = 0.2727...
        # Embedding similarity: 0.0, Scaled embedding similarity: 0.0
        # Combined: (1-0.5)*0.2727 + 0.5*0 = 0.13635
        sku1_sku2_score = next(s[1] for s in subs.get("SKU1", []) if s[0] == "SKU2")
        self.assertAlmostEqual(sku1_sku2_score, (3.0 / self.max_stat_score) * (1.0 - 0.5) + 0.0 * 0.5)


    def test_find_substitutes_with_embeddings_zero_weight(self):
        subs = find_substitutes(self.detailed_results, self.sku_embeddings, embedding_weight=0.0, k=2)
        # Scores should be based purely on statistical effects (normalized)
        # SKU1 -> SKU2: stat_score = (2.0+1.0)/11.0 = 0.2727...; emb_weight=0 -> score = 0.2727...
        sku1_subs = subs.get("SKU1", [])
        self.assertTrue(any(s[0] == "SKU2" for s in sku1_subs))
        sku1_sku2_score = next(s[1] for s in sku1_subs if s[0] == "SKU2")
        self.assertAlmostEqual(sku1_sku2_score, (2.0 + 1.0) / self.max_stat_score)
        
        sku1_sku2_details = next(s[2] for s in sku1_subs if s[0] == "SKU2")
        # Similarity should be calculated and stored even if weight is 0
        # SKU1: [1,0,0], SKU2: [0.9,0.1,0] -> dot=0.9, norm1=1, norm2=sqrt(0.81+0.01)=sqrt(0.82)
        # sim = 0.9 / sqrt(0.82) approx 0.9 / 0.9055 = 0.9939
        expected_raw_sim_s1_s2 = 0.9 / np.sqrt(0.82)
        expected_scaled_sim_s1_s2 = (expected_raw_sim_s1_s2 + 1.0) / 2.0
        self.assertAlmostEqual(sku1_sku2_details['embedding_similarity_score'], expected_raw_sim_s1_s2)
        self.assertAlmostEqual(sku1_sku2_details['scaled_embedding_similarity_score'], expected_scaled_sim_s1_s2)


    def test_find_substitutes_with_embeddings_full_weight(self):
        subs = find_substitutes(self.detailed_results, self.sku_embeddings, embedding_weight=1.0, k=2)
        # Scores should be based purely on scaled embedding similarity
        
        # SKU1 -> SKU2: Raw sim = 0.9939, Scaled sim = (0.9939+1)/2 = 0.99695
        # SKU1 -> SKU3: Raw sim = 0 (orthogonal), Scaled sim = (0+1)/2 = 0.5
        # SKU1 -> SKU4: No significance, but emb_weight=1.0 means score is scaled_similarity.
        # SKU1: [1,0,0], SKU4: [0.5,0.5,0.5]. dot=0.5. norm1=1. norm4=sqrt(0.75)
        # raw_sim_s1_s4 = 0.5 / sqrt(0.75) = 0.5 / 0.866 = 0.57735
        # scaled_sim_s1_s4 = (0.57735 + 1) / 2 = 0.788675
        
        sku1_subs = subs.get("SKU1", [])
        sku1_sku2_score = next(s[1] for s in sku1_subs if s[0] == "SKU2")
        sku1_sku4_score = next(s[1] for s in sku1_subs if s[0] == "SKU4")

        expected_raw_sim_s1_s2 = 0.9 / np.sqrt(0.82)
        expected_scaled_sim_s1_s2 = (expected_raw_sim_s1_s2 + 1.0) / 2.0
        self.assertAlmostEqual(sku1_sku2_score, expected_scaled_sim_s1_s2)

        expected_raw_sim_s1_s4 = 0.5 / np.sqrt(0.75)
        expected_scaled_sim_s1_s4 = (expected_raw_sim_s1_s4 + 1.0) / 2.0
        self.assertAlmostEqual(sku1_sku4_score, expected_scaled_sim_s1_s4)
        
        # Check ranking, SKU2 should be higher than SKU4 for SKU1
        self.assertTrue(sku1_sku2_score > sku1_sku4_score)


    def test_find_substitutes_with_embeddings_mixed_weight(self):
        embedding_weight = 0.5
        subs = find_substitutes(self.detailed_results, self.sku_embeddings, embedding_weight=embedding_weight, k=3)
        
        # SKU1 -> SKU2:
        norm_stat_s1_s2 = (2.0 + 1.0) / self.max_stat_score # approx 0.2727
        raw_sim_s1_s2 = calculate_embedding_similarity(self.sku_embeddings["SKU1"], self.sku_embeddings["SKU2"]) # approx 0.9939
        scaled_sim_s1_s2 = (raw_sim_s1_s2 + 1.0) / 2.0 # approx 0.99695
        expected_score_s1_s2 = (1 - embedding_weight) * norm_stat_s1_s2 + embedding_weight * scaled_sim_s1_s2
        
        # SKU1 -> SKU4: (No stat significance, so norm_stat_s1_s4 should be 0 if require_significance=True)
        # The logic for require_significance=True and sig_count=0:
        # if embedding_weight == 1.0, score is scaled_sim, else 0.0.
        # Here embedding_weight is 0.5, so score should be 0 if require_significance=True.
        # Let's test with require_significance=False for this specific calculation to see mixed effect.
        
        subs_no_req_sig = find_substitutes(self.detailed_results, self.sku_embeddings, 
                                           embedding_weight=embedding_weight, k=3, require_significance=False)
        
        # For SKU1 -> SKU4 (no stat significance, require_significance=False)
        # oos_effect = 0.1 (not sig), price_effect = 0.05 (not sig) -> stat_contribution = 0.1 + 0.05 = 0.15
        norm_stat_s1_s4_no_sig = (0.1 + 0.05) / self.max_stat_score # approx 0.0136
        raw_sim_s1_s4 = calculate_embedding_similarity(self.sku_embeddings["SKU1"], self.sku_embeddings["SKU4"]) # approx 0.57735
        scaled_sim_s1_s4 = (raw_sim_s1_s4 + 1.0) / 2.0 # approx 0.788675
        expected_score_s1_s4_no_sig = (1-embedding_weight) * norm_stat_s1_s4_no_sig + embedding_weight * scaled_sim_s1_s4

        sku1_subs = subs.get("SKU1", [])
        sku1_sku2_score = next(s[1] for s in sku1_subs if s[0] == "SKU2")
        self.assertAlmostEqual(sku1_sku2_score, expected_score_s1_s2)

        sku1_subs_no_req_sig = subs_no_req_sig.get("SKU1", [])
        sku1_sku4_score_no_sig = next(s[1] for s in sku1_subs_no_req_sig if s[0] == "SKU4")
        self.assertAlmostEqual(sku1_sku4_score_no_sig, expected_score_s1_s4_no_sig)
        
        # Check that with require_significance=True, SKU1->SKU4 score is 0.0 (because weight is not 1.0)
        sku1_sku4_score_req_sig = next((s[1] for s in sku1_subs if s[0] == "SKU4"), 0.0) # Get score or 0.0 if not found
        self.assertAlmostEqual(sku1_sku4_score_req_sig, 0.0)


    def test_find_substitutes_missing_embeddings_for_some_items(self):
        # SKU5 embedding is missing in self.sku_embeddings
        embedding_weight = 0.5
        subs = find_substitutes(self.detailed_results, self.sku_embeddings, embedding_weight=embedding_weight, k=1)
        
        # SKU4 -> SKU5
        # Stat for SKU4->SKU5: (3.0+1.0) / 11.0 = 4.0/11.0 = 0.3636
        # Embedding for SKU5 is missing, so scaled_similarity_score will be 0.0 (raw will be 0.0, scaled (0+1)/2=0.5 if one is present, but calc_embed_sim returns 0 if any is None or empty)
        # The current calculate_embedding_similarity returns 0 if an embedding is None.
        # find_substitutes initializes scaled_similarity_score = 0.0. If get() returns None for an embedding, it won't call calc_embed_sim.
        # So scaled_similarity_score remains 0.0 for SKU4->SKU5 pair.

        sku4_subs = subs.get("SKU4", [])
        self.assertTrue(any(s[0] == "SKU5" for s in sku4_subs))
        sku4_sku5_score = next(s[1] for s in sku4_subs if s[0] == "SKU5")
        
        norm_stat_s4_s5 = (3.0 + 1.0) / self.max_stat_score
        expected_score_s4_s5 = (1 - embedding_weight) * norm_stat_s4_s5 + embedding_weight * 0.0 # scaled_similarity is 0.0
        self.assertAlmostEqual(sku4_sku5_score, expected_score_s4_s5)

        sku4_sku5_details = next(s[2] for s in sku4_subs if s[0] == "SKU5")
        self.assertAlmostEqual(sku4_sku5_details['embedding_similarity_score'], 0.0)
        self.assertAlmostEqual(sku4_sku5_details['scaled_embedding_similarity_score'], 0.0)

    def test_find_substitutes_require_significance_true_full_embedding_weight(self):
        # Test case where require_significance=True, stat effects are not significant,
        # but embedding_weight=1.0, so score should be based on similarity.
        # SKU1 -> SKU4 (no significant stat effects)
        subs = find_substitutes(self.detailed_results, self.sku_embeddings, embedding_weight=1.0, k=3, require_significance=True)
        
        sku1_subs = subs.get("SKU1", [])
        sku1_sku4_item = next((s for s in sku1_subs if s[0] == "SKU4"), None)
        self.assertIsNotNone(sku1_sku4_item, "SKU4 should be a potential substitute for SKU1")

        raw_sim_s1_s4 = calculate_embedding_similarity(self.sku_embeddings["SKU1"], self.sku_embeddings["SKU4"])
        scaled_sim_s1_s4 = (raw_sim_s1_s4 + 1.0) / 2.0
        
        self.assertAlmostEqual(sku1_sku4_item[1], scaled_sim_s1_s4) # Score should be scaled similarity
        self.assertAlmostEqual(sku1_sku4_item[2]['embedding_similarity_score'], raw_sim_s1_s4)
        self.assertAlmostEqual(sku1_sku4_item[2]['scaled_embedding_similarity_score'], scaled_sim_s1_s4)


class TestExportToCSV(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = self.temp_dir.name
        # Sample substitutes_dict
        self.substitutes_dict_sample = {
            "SKU1": [
                ("SKU2", 0.75, {
                    'oos_effect': 2.0, 'price_effect': 1.0, 'promo_effect': 0.1,
                    'relationship_type': 'Substitute',
                    'embedding_similarity_score': 0.9,
                    'scaled_embedding_similarity_score': 0.95
                }),
                ("SKU3", 0.35, {
                    'oos_effect': 0.5, 'price_effect': -0.2, 'promo_effect': 0.0,
                    'relationship_type': 'Complement',
                    'embedding_similarity_score': -0.4,
                    'scaled_embedding_similarity_score': 0.3
                })
            ],
            "SKU4": [] # No substitutes
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_export_csv_with_embedding_scores(self):
        csv_path = export_to_csv(self.substitutes_dict_sample, self.output_dir)
        self.assertTrue(os.path.exists(csv_path))

        df = pd.read_csv(csv_path)
        
        expected_columns = [
            'primary_item', 'substitute_item', 'combined_score',
            'oos_effect', 'price_effect', 'promo_effect', 'relationship_type',
            'embedding_similarity_score', 'scaled_embedding_similarity_score'
        ]
        self.assertListEqual(list(df.columns), expected_columns)
        
        self.assertEqual(len(df), 2) # Two substitute pairs in the sample
        
        # Check values for SKU1 -> SKU2
        row_s1_s2 = df[(df['primary_item'] == "SKU1") & (df['substitute_item'] == "SKU2")]
        self.assertAlmostEqual(row_s1_s2['combined_score'].iloc[0], 0.75)
        self.assertAlmostEqual(row_s1_s2['embedding_similarity_score'].iloc[0], 0.9)
        self.assertAlmostEqual(row_s1_s2['scaled_embedding_similarity_score'].iloc[0], 0.95)
        self.assertEqual(row_s1_s2['relationship_type'].iloc[0], 'Substitute')

        # Check values for SKU1 -> SKU3
        row_s1_s3 = df[(df['primary_item'] == "SKU1") & (df['substitute_item'] == "SKU3")]
        self.assertAlmostEqual(row_s1_s3['combined_score'].iloc[0], 0.35)
        self.assertAlmostEqual(row_s1_s3['embedding_similarity_score'].iloc[0], -0.4)
        self.assertAlmostEqual(row_s1_s3['scaled_embedding_similarity_score'].iloc[0], 0.3)
        self.assertEqual(row_s1_s3['relationship_type'].iloc[0], 'Complement')


if __name__ == '__main__':
    unittest.main()
