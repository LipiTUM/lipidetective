import os
import re
import torch
import pathlib
import torch.nn.functional as F
from statistics import mean

from src.lipidetective.helpers.utils import read_yaml, is_lipid_class_with_slash


class LipidLibrary:
    def __init__(self):
        cwd = pathlib.Path(__file__).parent.parent.resolve()

        self.molecular_lipid_species = read_yaml(os.path.join(cwd, 'lipid_info/molecular_lipid_species.yaml'))
        self.sum_lipid_species = read_yaml(os.path.join(cwd, 'lipid_info/sum_lipid_species.yaml'))
        self.headgroups = read_yaml(os.path.join(cwd, 'lipid_info/headgroups.yaml'))
        self.side_chains = read_yaml(os.path.join(cwd, 'lipid_info/sidechains.yaml'))
        self.tokens = read_yaml(os.path.join(cwd, 'lipid_info/lipid_components_tokens.yaml'))
        self.tokens_inv = {v: k for k, v in self.tokens.items()}
        self.nr_tokens = len(self.tokens)

        self.side_chains_to_mass = dict([(key, value['mono_mass']) for key, value in self.side_chains.items()])

        self.mean_mass_headgroup = mean(self.headgroups.values())
        self.mean_mass_side_chains = mean(self.side_chains_to_mass.values())
        self.max_mass_headgroup = max(self.headgroups.values()) - self.mean_mass_headgroup
        self.max_mass_side_chains = max(self.side_chains_to_mass.values()) - self.mean_mass_side_chains

        self.max_precursor_mass = 0
        sum_precursor_mass = 0
        count_precursors = 0

        for species in self.sum_lipid_species.values():
            if 'pos' in species:
                pos_adducts = species['pos']
                for precursor in pos_adducts.values():
                    if precursor > self.max_precursor_mass:
                        self.max_precursor_mass = precursor
                    sum_precursor_mass += precursor
                    count_precursors += 1

            if 'neg' in species:
                neg_adducts = species['neg']
                for precursor in neg_adducts.values():
                    if precursor > self.max_precursor_mass:
                        self.max_precursor_mass = precursor
                    sum_precursor_mass += precursor
                    count_precursors += 1

        self.mean_precursor_mass = sum_precursor_mass / count_precursors
        self.max_precursor_mass_norm = self.max_precursor_mass - self.mean_precursor_mass

        self.headgroups_mass_norm = {key: ((value - self.mean_mass_headgroup) / self.max_mass_headgroup) for key, value in self.headgroups.items()}
        self.side_chains_mass_norm = {key: ((value - self.mean_mass_side_chains) / self.max_mass_side_chains) for key, value in self.side_chains_to_mass.items()}

        self.lipid_class_re = re.compile(r'^[a-zA-Z]+[_a-zA-Z0-9]*(\s+|$)')
        self.fatty_acids_re = re.compile(r'[0-9]{1,2}:[0-9]{1,2}')
        self.bond_types_re = re.compile(r'(?<= )[a-zA-Z]{1,2}[\-]{0,1}(?=[1-9]+)')
        self.bond_types_2_re = re.compile(r'(?<=_)[a-zA-Z]{1,2}[\-]{0,1}(?=[1-9]+)')
        self.functional_groups_re_1 = re.compile(r'(?<= [0-9]{2}:[0-9]{1});[a-zA-Z1-9]+(?=[_/])')
        self.functional_groups_re_1_2 = re.compile(r'(?<= [0-9]{1}:[0-9]{1});[a-zA-Z1-9]+(?=[_/])')
        self.functional_groups_re_2 = re.compile(r';[a-zA-Z1-9]+(?=$)')
        self.functional_groups_re_3 = re.compile(r';[a-zA-Z1-9]+(?=[;])')
        self.functional_groups_re_4 = re.compile(r'(?<=_[0-9]{2}:[0-9]{1});[a-zA-Z1-9]+(?=[_/])')
        self.adduct_re = re.compile(r' \[M[+\-]\S+][+\-](?=<EOS>)')

    def get_regression_label(self, lipid_species: str):
        species_info = self.molecular_lipid_species[lipid_species].copy()
        species_info['molecular_lipid_species'] = lipid_species

        headgroup_mass = self.headgroups_mass_norm[species_info['headgroup']]
        sc_1_mass = self.side_chains_mass_norm[species_info['fatty_acid_sn1']]
        sc_2_mass = self.side_chains_mass_norm[species_info['fatty_acid_sn2']]

        label = torch.tensor([headgroup_mass, sc_1_mass, sc_2_mass])

        return label, species_info

    def get_transformer_label(self, lipid_species: str, adduct: str, output_seq_length: int):
        if lipid_species:
            lipid_name_components = self.parse_lipid_species_components(lipid_species)
            name_tokens = [self.tokens[letter] for letter in lipid_name_components]

            adduct_tokens = self.tokens[' ' + adduct]

            token_ids = [self.tokens["<SOS>"]] + name_tokens + [adduct_tokens] + [self.tokens["<EOS>"]]
        else:
            lipid_name_components = ['']
            token_ids = [self.tokens["<SOS>"], self.tokens["<EOS>"]]

        token_tensor = torch.IntTensor(token_ids)

        if token_tensor.size()[0] < output_seq_length:
            nr_tokens_to_append = output_seq_length - token_tensor.size()[0]
            token_tensor = F.pad(input=token_tensor, pad=(0, nr_tokens_to_append), mode='constant', value=0)

        species_info = {}
        species_info['molecular_lipid_species'] = lipid_species
        species_info['adduct'] = adduct
        species_info['lipid_class'] = lipid_name_components[0]

        return token_tensor, species_info

    def parse_lipid_species_components(self, lipid_species: str):
        lipid_class = self.lipid_class_re.match(lipid_species).group()
        fatty_acids = self.fatty_acids_re.findall(lipid_species)
        bond_types = self.bond_types_re.findall(lipid_species)
        bond_types_2 = self.bond_types_2_re.findall(lipid_species)
        functional_groups_1 = self.functional_groups_re_1.findall(lipid_species)
        functional_groups_1_2 = self.functional_groups_re_1_2.findall(lipid_species)
        functional_groups_2 = self.functional_groups_re_2.findall(lipid_species)
        functional_groups_3 = self.functional_groups_re_3.findall(lipid_species)
        functional_groups_4 = self.functional_groups_re_4.findall(lipid_species)

        lipid_name_components = [lipid_class]

        if bond_types:
            lipid_name_components.append(bond_types[0])

        for idx, fatty_acid in enumerate(fatty_acids):
            if idx > 0:
                if is_lipid_class_with_slash(lipid_species):
                    lipid_name_components.append('/')
                else:
                    lipid_name_components.append('_')

            if idx == 1 and bond_types_2:
                lipid_name_components.append(bond_types_2[0])

            lipid_name_components.append(fatty_acid)

            if idx == 0 and functional_groups_1:
                lipid_name_components.append(functional_groups_1[0])
            elif idx == 0 and functional_groups_1_2:
                lipid_name_components.append(functional_groups_1_2[0])

            if idx == 1 and functional_groups_4:
                lipid_name_components.append(functional_groups_4[0])

        if functional_groups_3:
            for group in functional_groups_3:
                lipid_name_components.append(group)

        if functional_groups_2:
            lipid_name_components.append(functional_groups_2[0])

        return lipid_name_components

    def get_lipid_species_components(self, lipid_species: str):
        lipid_class = self.lipid_class_re.match(lipid_species)
        if lipid_class:
            lipid_class = lipid_class.group()
        else:
            lipid_class = ''
        lipid_class = lipid_class[0] if lipid_class else lipid_class
        fatty_acids = self.fatty_acids_re.findall(lipid_species)
        bond_types = self.bond_types_re.findall(lipid_species)
        bond_types_2 = self.bond_types_2_re.findall(lipid_species)
        functional_groups_1 = self.functional_groups_re_1.findall(lipid_species)
        functional_groups_2 = self.functional_groups_re_2.findall(lipid_species)
        functional_groups_3 = self.functional_groups_re_3.findall(lipid_species)

        return lipid_class, fatty_acids, bond_types, bond_types_2, functional_groups_1, functional_groups_2, functional_groups_3

    def normalize_precursor_mass(self, precursor_mass):
        return (precursor_mass - self.mean_precursor_mass) / self.max_precursor_mass_norm

    def translate_tokens_to_name(self, tokens):
        name = [self.tokens_inv[token.item()] for token in tokens]
        return name

    def custom_accuracy_scoring(self, prediction, label):
        pred_class, pred_fa, pred_bond1, pred_bond2, pred_func_group_1, pred_func_group_2, pred_func_group_3 = self.get_lipid_species_components(
            prediction)
        label_class, label_fa, label_bond1, label_bond2, label_func_group_1, label_func_group_2, label_func_group_3 = self.get_lipid_species_components(
            label)

        adduct_pred = self.adduct_re.findall(prediction)
        adduct_label = self.adduct_re.findall(label)

        nr_total = len(label_fa)
        nr_correct = 0

        for fatty_acid in label_fa:
            if fatty_acid in pred_fa:
                nr_correct += 1
                fa_idx = pred_fa.index(fatty_acid)
                del pred_fa[fa_idx]

        if label_class:
            nr_total += 1
            if pred_class == label_class:
                nr_correct += 1

        if label_bond1:
            nr_total += 1
            if label_bond1 == pred_bond1:
                nr_correct += 1

        if label_bond2:
            nr_total += 1
            if label_bond2 == pred_bond2:
                nr_correct += 1

        if label_func_group_1:
            nr_total += 1
            if label_func_group_1 == pred_func_group_1:
                nr_correct += 1

        if label_func_group_2:
            nr_total += 1
            if label_func_group_2 == pred_func_group_2:
                nr_correct += 1

        if label_func_group_3:
            nr_total += len(label_func_group_3)
            for group in label_func_group_3:
                if group in pred_func_group_3:
                    nr_correct += 1
                    group_idx = pred_func_group_3.index(group)
                    del pred_func_group_3[group_idx]

        if adduct_label:
            nr_total += 1
            if adduct_label == adduct_pred:
                nr_correct += 1

        if nr_total == 0:
            if prediction == label:
                nr_total = 1
                nr_correct = 1
            else:
                nr_total = 1

        return nr_correct / nr_total


