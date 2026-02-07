"""Tests for LipidLibrary class."""

import pytest
import torch


class TestLipidLibraryInitialization:
    """Tests for LipidLibrary initialization and loaded data."""

    def test_initialization_loads_yaml_files(self, lipid_library):
        """LipidLibrary should load all required YAML files on init."""
        assert lipid_library.molecular_lipid_species is not None
        assert lipid_library.sum_lipid_species is not None
        assert lipid_library.headgroups is not None
        assert lipid_library.side_chains is not None
        assert lipid_library.tokens is not None

    def test_tokens_are_invertible(self, lipid_library):
        """Token mappings should be invertible (token -> id -> token)."""
        for token, token_id in lipid_library.tokens.items():
            assert lipid_library.tokens_inv[token_id] == token

    def test_special_tokens_exist(self, lipid_library):
        """Special tokens SOS, EOS, PAD should exist in vocabulary."""
        assert "<SOS>" in lipid_library.tokens
        assert "<EOS>" in lipid_library.tokens
        assert "<PAD>" in lipid_library.tokens


class TestParseLipidSpeciesComponents:
    """Tests for parsing lipid nomenclature into components."""

    @pytest.mark.parametrize(
        "lipid_species,expected_class",
        [
            ("PC 16:0_18:1", "PC "),
            ("PE 18:0_18:2", "PE "),
            ("SM d18:1/16:0", "SM "),
            ("TG 16:0_18:1_18:2", "TG "),
            ("Cer d18:1/24:0", "Cer "),
            ("LPC 18:0", "LPC "),
        ],
    )
    def test_parses_lipid_class(self, lipid_library, lipid_species, expected_class):
        """Should correctly extract lipid class from nomenclature."""
        components = lipid_library.parse_lipid_species_components(lipid_species)
        assert components[0] == expected_class

    @pytest.mark.parametrize(
        "lipid_species,expected_fatty_acids",
        [
            ("PC 16:0_18:1", ["16:0", "18:1"]),
            ("PE 18:0_18:2", ["18:0", "18:2"]),
            ("TG 16:0_18:1_18:2", ["16:0", "18:1", "18:2"]),
            ("LPC 18:0", ["18:0"]),
        ],
    )
    def test_parses_fatty_acids(self, lipid_library, lipid_species, expected_fatty_acids):
        """Should correctly extract fatty acid chains."""
        components = lipid_library.parse_lipid_species_components(lipid_species)
        # Filter out non-fatty-acid components (class, separators)
        fatty_acids = [c for c in components if ":" in c and c[0].isdigit()]
        assert fatty_acids == expected_fatty_acids

    def test_parses_underscore_separator(self, lipid_library):
        """Should include underscore separator between fatty acids."""
        components = lipid_library.parse_lipid_species_components("PC 16:0_18:1")
        assert "_" in components

    def test_parses_slash_separator(self, lipid_library):
        """Should include slash separator for sn-position-specific lipids."""
        components = lipid_library.parse_lipid_species_components("SM d18:1/16:0")
        assert "/" in components

    def test_invalid_lipid_raises_error(self, lipid_library):
        """Should raise ValueError for unparseable lipid nomenclature."""
        with pytest.raises(ValueError, match="Could not parse lipid class"):
            lipid_library.parse_lipid_species_components("12345")


class TestGetTransformerLabel:
    """Tests for transformer label generation."""

    def test_returns_tensor_and_info(self, lipid_library):
        """Should return token tensor and species info dict."""
        token_tensor, species_info = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=20
        )
        assert isinstance(token_tensor, torch.Tensor)
        assert isinstance(species_info, dict)

    def test_tensor_starts_with_sos(self, lipid_library):
        """Token tensor should start with SOS token."""
        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=20
        )
        sos_id = lipid_library.tokens["<SOS>"]
        assert token_tensor[0].item() == sos_id

    def test_tensor_padded_to_output_length(self, lipid_library):
        """Token tensor should be padded to specified output sequence length."""
        output_seq_length = 25
        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=output_seq_length
        )
        assert token_tensor.shape[0] == output_seq_length

    def test_species_info_contains_metadata(self, lipid_library):
        """Species info should contain lipid species, adduct, and class."""
        _, species_info = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=20
        )
        assert species_info["molecular_lipid_species"] == "PC 16:0_18:1"
        assert species_info["adduct"] == "[M+H]+"
        assert "lipid_class" in species_info

    def test_empty_lipid_species_returns_minimal_tokens(self, lipid_library):
        """Empty lipid species should return SOS + EOS only."""
        token_tensor, _ = lipid_library.get_transformer_label("", "[M+H]+", output_seq_length=20)
        sos_id = lipid_library.tokens["<SOS>"]
        eos_id = lipid_library.tokens["<EOS>"]
        assert token_tensor[0].item() == sos_id
        assert token_tensor[1].item() == eos_id


class TestTranslateTokensToName:
    """Tests for token-to-name translation."""

    def test_roundtrip_conversion(self, lipid_library):
        """Converting to tokens and back should preserve the structure."""
        token_tensor, _ = lipid_library.get_transformer_label(
            "PC 16:0_18:1", "[M+H]+", output_seq_length=20
        )
        # Skip SOS token at index 0
        name_parts = lipid_library.translate_tokens_to_name(token_tensor[1:])
        name_str = "".join(name_parts).replace("<PAD>", "").replace("<EOS>", "")
        # Should reconstruct the lipid name with adduct
        assert "PC " in name_str
        assert "16:0" in name_str
        assert "18:1" in name_str


class TestCustomAccuracyScoring:
    """Tests for lipid-aware accuracy scoring."""

    def test_identical_prediction_scores_1(self, lipid_library):
        """Identical prediction and label should score 1.0."""
        lipid = "PC 16:0_18:1 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(lipid, lipid)
        assert score == 1.0

    def test_wrong_class_reduces_score(self, lipid_library):
        """Wrong lipid class should reduce accuracy score."""
        # PE vs PC - same fatty acids but different headgroup
        prediction = "PE 16:0_18:1 [M+H]+<EOS>"
        label = "PC 16:0_18:1 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score == 0.75  # 3/4: two fatty acids + adduct correct, class wrong

    def test_partial_fatty_acid_match(self, lipid_library):
        """Matching one of two fatty acids should give partial credit."""
        prediction = "PC 16:0_20:4 [M+H]+<EOS>"
        label = "PC 16:0_18:1 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        # Should get credit for class + one fatty acid
        assert 0 < score < 1.0

    def test_completely_wrong_scores_low(self, lipid_library):
        """Completely wrong prediction should score very low."""
        prediction = "TG 20:0_22:0_24:0 [M+H]+<EOS>"
        label = "PC 16:0_18:1 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score < 0.5


class TestGetRegressionLabel:
    """Tests for regression label generation."""

    def test_returns_tensor_and_info(self, lipid_library):
        """Should return 3-element tensor and species info."""
        # Use a lipid that exists in molecular_lipid_species
        sample_lipid = list(lipid_library.molecular_lipid_species.keys())[0]
        label, species_info = lipid_library.get_regression_label(sample_lipid)
        assert isinstance(label, torch.Tensor)
        assert label.shape == (3,)
        assert isinstance(species_info, dict)

    def test_label_values_are_normalized(self, lipid_library):
        """Regression labels should be normalized mass values."""
        sample_lipid = list(lipid_library.molecular_lipid_species.keys())[0]
        label, _ = lipid_library.get_regression_label(sample_lipid)
        # Normalized values should typically be in [-1, 1] range
        assert all(-5 <= v <= 5 for v in label.tolist())


class TestNormalizePrecursorMass:
    """Tests for precursor mass normalization."""

    def test_normalization_formula(self, lipid_library):
        """Normalization should follow (mass - mean) / max_norm formula."""
        test_mass = 800.0
        normalized = lipid_library.normalize_precursor_mass(test_mass)
        expected = (
            test_mass - lipid_library.mean_precursor_mass
        ) / lipid_library.max_precursor_mass_norm
        assert normalized == pytest.approx(expected)

    def test_mean_mass_normalizes_to_zero(self, lipid_library):
        """Mean precursor mass should normalize to approximately 0."""
        normalized = lipid_library.normalize_precursor_mass(lipid_library.mean_precursor_mass)
        assert normalized == pytest.approx(0.0, abs=1e-10)


class TestParseLipidSpeciesComponentsAdvanced:
    """Advanced tests for parsing complex lipid nomenclature."""

    def test_parses_bond_types_2(self, lipid_library):
        """Should parse secondary bond types (e.g., O- for ether linkage)."""
        # PE with ether linkage on second position: PE 18:0_O-16:0
        components = lipid_library.parse_lipid_species_components("PE 18:0_O-16:0")
        assert "O-" in components or any("O-" in str(c) for c in components)

    def test_parses_functional_groups_type_1(self, lipid_library):
        """Should parse functional groups after first fatty acid (;OH format)."""
        # e.g., PC 16:0;OH_18:1 or similar
        components = lipid_library.parse_lipid_species_components("PC 16:0;O_18:1")
        assert ";O" in components

    def test_parses_functional_groups_type_1_2_single_digit(self, lipid_library):
        """Should parse functional groups after single-digit carbon fatty acid."""
        # Line 156 coverage: functional_groups_re_1_2 with single digit (e.g., 8:0)
        components = lipid_library.parse_lipid_species_components("PC 8:0;O_18:1")
        assert ";O" in components

    def test_parses_functional_groups_type_2(self, lipid_library):
        """Should parse terminal functional groups."""
        # Functional group at end: PC 16:0_18:1;O
        components = lipid_library.parse_lipid_species_components("PC 16:0_18:1;O")
        assert ";O" in components

    def test_parses_functional_groups_type_3(self, lipid_library):
        """Should parse multiple functional groups with semicolons."""
        # Multiple functional groups: lipid;O;O format
        components = lipid_library.parse_lipid_species_components("PC 16:0_18:1;O;O")
        # Should have functional groups parsed
        assert components.count(";O") >= 1

    def test_parses_functional_groups_type_4(self, lipid_library):
        """Should parse functional group after second fatty acid before separator."""
        # Line 159 coverage: functional_groups_re_4
        components = lipid_library.parse_lipid_species_components("TG 16:0_18:1;O_20:4")
        assert ";O" in components


class TestGetLipidSpeciesComponents:
    """Tests for get_lipid_species_components method."""

    def test_returns_all_components(self, lipid_library):
        """Should return tuple of all component types."""
        result = lipid_library.get_lipid_species_components("PC 16:0_18:1")
        assert len(result) == 7
        lipid_class, fatty_acids, bond1, bond2, fg1, fg2, fg3 = result
        assert lipid_class == "PC"
        assert fatty_acids == ["16:0", "18:1"]

    def test_empty_lipid_returns_empty_components(self, lipid_library):
        """Should handle empty/invalid lipid gracefully."""
        result = lipid_library.get_lipid_species_components("")
        lipid_class, fatty_acids, _, _, _, _, _ = result
        assert lipid_class == ""
        assert fatty_acids == []

    def test_extracts_full_class_name(self, lipid_library):
        """Lipid class should be the full class name."""
        result = lipid_library.get_lipid_species_components("PC 16:0_18:1")
        assert result[0] == "PC"


class TestCustomAccuracyScoringAdvanced:
    """Advanced tests for custom accuracy scoring edge cases."""

    def test_bond_type_matching(self, lipid_library):
        """Should score bond types correctly."""
        # Same lipid with bond type info
        prediction = "PE O-16:0_18:1 [M+H]+<EOS>"
        label = "PE O-16:0_18:1 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score == 1.0

    def test_wrong_bond_type_reduces_score(self, lipid_library):
        """Wrong bond type should reduce score."""
        prediction = "PE P-16:0_18:1 [M+H]+<EOS>"
        label = "PE O-16:0_18:1 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score < 1.0

    def test_bond_type_2_matching(self, lipid_library):
        """Should score bond type 2 (on second fatty acid) correctly."""
        # Lines 243-245 coverage: bond_types_2 matching
        prediction = "PE 18:0_O-16:0 [M+H]+<EOS>"
        label = "PE 18:0_O-16:0 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score == 1.0

    def test_wrong_bond_type_2_reduces_score(self, lipid_library):
        """Wrong bond type 2 should reduce score."""
        prediction = "PE 18:0_P-16:0 [M+H]+<EOS>"
        label = "PE 18:0_O-16:0 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score < 1.0

    def test_functional_group_1_matching(self, lipid_library):
        """Should score functional group type 1 (after first FA) correctly."""
        # Lines 248-250 coverage: func_group_1 matching
        prediction = "PC 16:0;O_18:1 [M+H]+<EOS>"
        label = "PC 16:0;O_18:1 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score == 1.0

    def test_wrong_functional_group_1_reduces_score(self, lipid_library):
        """Wrong functional group type 1 should reduce score."""
        prediction = "PC 16:0_18:1 [M+H]+<EOS>"
        label = "PC 16:0;O_18:1 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score < 1.0

    def test_functional_group_matching(self, lipid_library):
        """Should score functional groups correctly."""
        prediction = "PC 16:0_18:1;O [M+H]+<EOS>"
        label = "PC 16:0_18:1;O [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score == 1.0

    def test_wrong_functional_group_reduces_score(self, lipid_library):
        """Wrong functional group should reduce score."""
        prediction = "PC 16:0_18:1 [M+H]+<EOS>"
        label = "PC 16:0_18:1;O [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score < 1.0

    def test_adduct_matching(self, lipid_library):
        """Should score adduct correctly."""
        prediction = "PC 16:0_18:1 [M+H]+<EOS>"
        label = "PC 16:0_18:1 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score == 1.0

    def test_wrong_adduct_reduces_score(self, lipid_library):
        """Wrong adduct should reduce score."""
        prediction = "PC 16:0_18:1 [M+Na]+<EOS>"
        label = "PC 16:0_18:1 [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score < 1.0

    def test_empty_prediction_and_label_equal(self, lipid_library):
        """Empty prediction matching empty label should score 1.0."""
        score = lipid_library.custom_accuracy_scoring("", "")
        assert score == 1.0

    def test_empty_prediction_not_matching_label(self, lipid_library):
        """Empty prediction vs non-empty label scores 0."""
        score = lipid_library.custom_accuracy_scoring("", "PC 16:0")
        # When nr_total is 0 and prediction != label, returns 0
        assert score == 0.0

    def test_unparseable_but_different_strings(self, lipid_library):
        """Different unparseable strings should score 0."""
        # Line 275 coverage: nr_total == 0, prediction != label
        score = lipid_library.custom_accuracy_scoring("abc", "xyz")
        assert score == 0.0

    def test_multiple_functional_groups_type_3(self, lipid_library):
        """Should handle multiple functional groups of type 3."""
        prediction = "PC 16:0_18:1;O;OH [M+H]+<EOS>"
        label = "PC 16:0_18:1;O;OH [M+H]+<EOS>"
        score = lipid_library.custom_accuracy_scoring(prediction, label)
        assert score == 1.0
