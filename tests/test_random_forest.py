"""Unit tests for random_forest.py."""

import os
import tempfile


class TestCheckClassificationAccuracy:
    """Tests for check_classification_accuracy method."""

    def test_returns_true_for_exact_match(self, rf_instance):
        """Should return True when prediction equals label."""
        result = rf_instance.check_classification_accuracy("PC 34:1", "PC 34:1")
        assert result is True

    def test_returns_false_for_mismatch(self, rf_instance):
        """Should return False when prediction differs from label."""
        result = rf_instance.check_classification_accuracy("PC 34:1", "PE 34:1")
        assert result is False

    def test_handles_list_inputs(self, rf_instance):
        """Should handle list inputs correctly."""
        pred = ["PC", "16:0", "18:1"]
        label = ["PC", "16:0", "18:1"]
        result = rf_instance.check_classification_accuracy(pred, label)
        assert result is True

    def test_list_mismatch(self, rf_instance):
        """Should return False for mismatched lists."""
        pred = ["PC", "16:0", "18:1"]
        label = ["PE", "16:0", "18:1"]
        result = rf_instance.check_classification_accuracy(pred, label)
        assert result is False


class TestCheckRegressionAccuracy:
    """Tests for check_regression_accuracy method."""

    def test_returns_true_for_close_values(self, rf_instance):
        """Should return True when all differences are < 0.1."""
        pred = [100.05, 200.01, 300.09]
        label = [100.0, 200.0, 300.0]
        result = rf_instance.check_regression_accuracy(pred, label)
        assert result is True

    def test_returns_false_for_one_large_diff(self, rf_instance):
        """Should return False when any difference is >= 0.1."""
        pred = [100.05, 200.15, 300.09]  # 200.15 - 200.0 = 0.15 >= 0.1
        label = [100.0, 200.0, 300.0]
        result = rf_instance.check_regression_accuracy(pred, label)
        assert result is False

    def test_above_boundary_value(self, rf_instance):
        """Should return False when difference is >= 0.1."""
        pred = [100.11]  # 0.11 > 0.1
        label = [100.0]
        result = rf_instance.check_regression_accuracy(pred, label)
        assert result is False

    def test_just_under_boundary(self, rf_instance):
        """Should return True when difference is just under 0.1."""
        pred = [100.099]
        label = [100.0]
        result = rf_instance.check_regression_accuracy(pred, label)
        assert result is True


class TestCalculateAccuracy:
    """Tests for calculate_accuracy method."""

    def test_classification_accuracy_calculation(self, rf_instance, caplog):
        """Should calculate accuracy for classification predictions."""
        import logging

        predictions = ["PC 34:1", "PE 36:2", "PC 34:1"]
        labels = ["PC 34:1", "PE 36:2", "PS 38:4"]

        with caplog.at_level(logging.INFO):
            result = rf_instance.calculate_accuracy(
                predictions, labels, "Test Model", "classification"
            )

        assert "Correct: 2" in result
        assert "Total:3" in result
        assert "Test Model" in caplog.text

    def test_regression_accuracy_calculation(self, rf_instance, caplog):
        """Should calculate accuracy for regression predictions."""
        import logging

        predictions = [[100.05, 200.01, 300.09], [100.5, 200.5, 300.5]]
        labels = [[100.0, 200.0, 300.0], [100.0, 200.0, 300.0]]

        with caplog.at_level(logging.INFO):
            result = rf_instance.calculate_accuracy(predictions, labels, "Regressor", "regression")

        assert "Correct: 1" in result  # Only first is within tolerance
        assert "Total:2" in result
        assert "Regressor" in caplog.text


class TestWriteOutputToFile:
    """Tests for write_output_to_file method."""

    def test_writes_statistics_to_file(self, rf_instance):
        """Should write statistics to output file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rf_instance.config["files"]["output"] = tmp_dir
            statistics = {"single_classifier": "Test statistics content\nAccuracy: 95%"}

            rf_instance.write_output_to_file(statistics)

            expected_file = os.path.join(
                tmp_dir, "random_forest_single_classifier_prediction_summary.txt"
            )
            assert os.path.exists(expected_file)
            with open(expected_file) as f:
                content = f.read()
            assert "Test statistics content" in content
            assert "Accuracy: 95%" in content

    def test_writes_multiple_model_statistics(self, rf_instance):
        """Should write statistics for multiple models."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            rf_instance.config["files"]["output"] = tmp_dir
            statistics = {
                "model_a": "Model A stats",
                "model_b": "Model B stats",
            }

            rf_instance.write_output_to_file(statistics)

            assert os.path.exists(
                os.path.join(tmp_dir, "random_forest_model_a_prediction_summary.txt")
            )
            assert os.path.exists(
                os.path.join(tmp_dir, "random_forest_model_b_prediction_summary.txt")
            )


class TestExtractFeaturesAndLabels:
    """Tests for extract_features_and_labels method."""

    def test_extracts_features_correctly(self, rf_instance):
        """Should extract features from dataset rows."""
        dataset = [
            [[100.1, 200.2], "PC 34:1", "PC", "16:0", "18:1", 184.07, 256.24, 282.26],
            [[150.1, 250.2], "PE 36:2", "PE", "18:1", "18:1", 141.02, 282.26, 282.26],
        ]

        features, labels = rf_instance.extract_features_and_labels(dataset)

        assert len(features) == 2
        assert len(labels) == 2
        # Features should be the first element of each row
        assert all(isinstance(f, list) for f in features)
        # Labels should be everything after first element
        assert len(labels[0]) == 7  # lipid_species + hg + fa1 + fa2 + 3 masses


class TestUseSingleClassifier:
    """Tests for use_single_classifier method."""

    def test_trains_and_predicts(self, rf_instance):
        """Should train classifier and return predictions."""
        # Create simple training data
        train_features = [[1, 2], [2, 3], [3, 4], [4, 5]]
        train_labels = [
            ["PC 34:1", "PC", "16:0", "18:1"],
            ["PC 34:1", "PC", "16:0", "18:1"],
            ["PE 36:2", "PE", "18:1", "18:1"],
            ["PE 36:2", "PE", "18:1", "18:1"],
        ]
        test_features = [[1.5, 2.5], [3.5, 4.5]]

        predictions, classifier = rf_instance.use_single_classifier(
            train_features, train_labels, test_features
        )

        assert len(predictions) == 2
        assert classifier is not None


class TestUseTripleClassifier:
    """Tests for use_triple_classifier method."""

    def test_trains_three_classifiers(self, rf_instance):
        """Should train and return predictions from three classifiers."""
        train_features = [[1, 2], [2, 3], [3, 4], [4, 5]]
        train_labels = [
            ["PC 34:1", "PC", "16:0", "18:1"],
            ["PC 34:1", "PC", "16:0", "18:1"],
            ["PE 36:2", "PE", "18:1", "18:1"],
            ["PE 36:2", "PE", "18:1", "18:1"],
        ]
        test_features = [[1.5, 2.5], [3.5, 4.5]]

        predictions, clf_hg, clf_fa1, clf_fa2 = rf_instance.use_triple_classifier(
            train_features, train_labels, test_features
        )

        assert len(predictions) == 2
        assert len(predictions[0]) == 3  # hg, fa1, fa2 predictions
        assert clf_hg is not None
        assert clf_fa1 is not None
        assert clf_fa2 is not None


class TestUseTripleRegressor:
    """Tests for use_triple_regressor method."""

    def test_trains_three_regressors(self, rf_instance):
        """Should train and return predictions from three regressors."""
        train_features = [[1, 2], [2, 3], [3, 4], [4, 5]]
        train_labels = [
            ["PC 34:1", "PC", "16:0", "18:1", 184.07, 256.24, 282.26],
            ["PC 34:1", "PC", "16:0", "18:1", 184.07, 256.24, 282.26],
            ["PE 36:2", "PE", "18:1", "18:1", 141.02, 282.26, 282.26],
            ["PE 36:2", "PE", "18:1", "18:1", 141.02, 282.26, 282.26],
        ]
        test_features = [[1.5, 2.5], [3.5, 4.5]]

        predictions, reg_hg, reg_fa1, reg_fa2 = rf_instance.use_triple_regressor(
            train_features, train_labels, test_features
        )

        assert len(predictions) == 2
        assert len(predictions[0]) == 3  # hg_mass, fa1_mass, fa2_mass predictions
        assert reg_hg is not None
        assert reg_fa1 is not None
        assert reg_fa2 is not None
