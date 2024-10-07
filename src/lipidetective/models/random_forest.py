import logging
import argparse
import os
import h5py
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils import shuffle
from sklearn import tree
from sklearn.model_selection import train_test_split

from src.lipidetective.helpers.utils import read_yaml


class RandomForest:
    def __init__(self, config: dict):
        self.config = config
        current_working_dir = os.getcwd()
        self.lipid_species_reference = read_yaml(os.path.join(current_working_dir, "lipid_info/molecular_lipid_species.yaml"))
        self.headgroup_reference = read_yaml(os.path.join(current_working_dir, "lipid_info/headgroups.yaml"))
        self.fatty_acid_reference = read_yaml(os.path.join(current_working_dir, "lipid_info/sidechains.yaml"))

        self.lipid_species = list(self.lipid_species_reference.keys())
        self.lipid_species_indices = [i for i in range(len(self.lipid_species))]
        self.lipid_species_key = dict(zip(self.lipid_species, self.lipid_species_indices))
        self.lipid_species_key_reverse = dict(zip(self.lipid_species_indices, self.lipid_species))

    def run(self):
        train_features, train_labels, test_features, test_labels = self.prepare_data()

        # Separate test labels for different models
        test_labels_lipid_names = [row[0] for row in test_labels]
        test_labels_lipid_components = [row[1:4] for row in test_labels]
        test_labels_lipid_masses = [row[4:] for row in test_labels]

        if self.config['random_forest']['type'] == 'single_classifier':
            # Single classifier prediction
            single_classifier_predictions, single_classifier = self.use_single_classifier(train_features, train_labels,
                                                                                     test_features)
            prediction_statistics_single = self.calculate_accuracy(single_classifier_predictions, test_labels_lipid_names,
                                                              "Single Classifier", "classification")
            self.plot_decision_tree(single_classifier.estimators_[0], 'random_forest_single_classifier_tree_1.png')

            self.write_output_to_file({'single_classifier': prediction_statistics_single})

        elif self.config['random_forest']['type'] == 'triple_classifier':
            # Triple classifier prediction
            triple_classifier_predictions, triple_classifier_hg, triple_classifier_fa1, triple_classifier_fa2 = self.use_triple_classifier(
                train_features, train_labels, test_features)
            prediction_statistics_triple = self.calculate_accuracy(triple_classifier_predictions, test_labels_lipid_components,
                                                              "Triple Classifier", "classification")
            self.plot_decision_tree(triple_classifier_hg.estimators_[0], 'random_forest_triple_classifier_hg_tree_1.png')
            self.plot_decision_tree(triple_classifier_fa1.estimators_[0], 'random_forest_triple_classifier_fa1_tree_1.png')
            self.plot_decision_tree(triple_classifier_fa2.estimators_[0], 'random_forest_triple_classifier_fa2_tree_1.png')

            self.write_output_to_file({'triple_classifier': prediction_statistics_triple})

        elif self.config['random_forest']['type'] == 'triple_regressor':
            # Triple regressor prediction
            triple_regressor_predictions, triple_regressor_hg, triple_regressor_fa1, triple_regressor_fa2 = self.use_triple_regressor(
                train_features, train_labels, test_features)
            prediction_statistics_triple_mass = self.calculate_accuracy(triple_regressor_predictions, test_labels_lipid_masses,
                                                                   "Triple Regressor", "regression")
            self.plot_decision_tree(triple_regressor_hg.estimators_[0], 'random_forest_triple_regressor_hg_tree_1.png')
            self.plot_decision_tree(triple_regressor_fa1.estimators_[0], 'random_forest_triple_regressor_fa1_tree_1.png')
            self.plot_decision_tree(triple_regressor_fa2.estimators_[0], 'random_forest_triple_regressor_fa2_tree_1.png')

            self.write_output_to_file({'triple_regressor': prediction_statistics_triple_mass})

    def get_spectrum_data(self, spectrum):
        mz, intensity = spectrum
        mz = list(mz)
        intensity = list(intensity)
        features = list(zip(mz, intensity))
        features.sort(key=lambda x: x[1], reverse=True)
        features = features[:30]
        features = [round(row[0], 3) for row in features]
        features.sort()

        lipid_species = spectrum.attrs['lipid_species']

        lipid_reference = self.lipid_species_reference[lipid_species]
        headgroup = lipid_reference['headgroup']
        fatty_acid_1 = lipid_reference['fatty_acid_sn1']
        fatty_acid_2 = lipid_reference['fatty_acid_sn2']

        headgroup_mass = self.headgroup_reference[headgroup]
        fatty_acid_1_mass = self.fatty_acid_reference[fatty_acid_1]['mono_mass']
        fatty_acid_2_mass = self.fatty_acid_reference[fatty_acid_2]['mono_mass']

        data_array = [features, lipid_species, headgroup, fatty_acid_1, fatty_acid_2, headgroup_mass, fatty_acid_1_mass,
                      fatty_acid_2_mass]

        return data_array

    def use_single_classifier(self, train_features, train_labels, test_features):
        classifier = RandomForestClassifier()

        train_labels_lipid_names = [row[0] for row in train_labels]

        classifier.fit(train_features, train_labels_lipid_names)

        return classifier.predict(test_features), classifier

    def plot_decision_tree(self, decision_tree, name_file):
        cn = self.lipid_species
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=300)
        tree.plot_tree(decision_tree, class_names=cn, filled=True, rounded=True, max_depth=3)
        fig.savefig(os.path.join(self.config['files']['output'], name_file))

    def use_triple_classifier(self, train_features, train_labels, test_features):
        train_labels_headgroup = [row[1] for row in train_labels]
        train_labels_fa1 = [row[2] for row in train_labels]
        train_labels_fa2 = [row[3] for row in train_labels]

        classifier_hg = RandomForestClassifier()
        classifier_hg.fit(train_features, train_labels_headgroup)
        classifier_hg_predictions = classifier_hg.predict(test_features)

        classifier_fa1 = RandomForestClassifier()
        classifier_fa1.fit(train_features, train_labels_fa1)
        classifier_fa1_predictions = classifier_fa1.predict(test_features)

        classifier_fa2 = RandomForestClassifier()
        classifier_fa2.fit(train_features, train_labels_fa2)
        classifier_fa2_predictions = classifier_fa2.predict(test_features)

        prediction_output = list(zip(classifier_hg_predictions, classifier_fa1_predictions, classifier_fa2_predictions))
        prediction_output = [list(item) for item in prediction_output]

        return prediction_output, classifier_hg, classifier_fa1, classifier_fa2

    def use_triple_regressor(self, train_features, train_labels, test_features):
        train_labels_headgroup_mass = [row[4] for row in train_labels]
        train_labels_fa1_mass = [row[5] for row in train_labels]
        train_labels_fa2_mass = [row[6] for row in train_labels]

        regressor_hg = RandomForestRegressor(max_depth=25)
        regressor_hg.fit(train_features, train_labels_headgroup_mass)
        regressor_hg_predictions = regressor_hg.predict(test_features)

        regressor_fa1 = RandomForestRegressor(max_depth=25)
        regressor_fa1.fit(train_features, train_labels_fa1_mass)
        regressor_fa1_predictions = regressor_fa1.predict(test_features)

        regressor_fa2 = RandomForestRegressor(max_depth=25)
        regressor_fa2.fit(train_features, train_labels_fa2_mass)
        regressor_fa2_predictions = regressor_fa2.predict(test_features)

        prediction_output = list(zip(regressor_hg_predictions, regressor_fa1_predictions, regressor_fa2_predictions))
        prediction_output = [list(item) for item in prediction_output]

        return prediction_output, regressor_hg, regressor_fa1, regressor_fa2

    def calculate_accuracy(self, prediction, labels, model, task):
        prediction_comparison = list(zip(prediction, labels))
        prediction_evaluation = []
        label_count = {}
        labels_correct = {}

        count_correct = 0
        total = 0

        for prediction in prediction_comparison:
            if task == 'classification':
                correct = self.check_classification_accuracy(prediction[0], prediction[1])
            else:
                correct = self.check_regression_accuracy(prediction[0], prediction[1])

            prediction_evaluation.append(f"{prediction} - {correct}")

            if correct:
                count_correct += 1
            total += 1

            prediction_str = str(prediction[0])

            if prediction_str in label_count:
                label_count[prediction_str] = label_count[prediction_str] + 1
            else:
                label_count[prediction_str] = 1

            if correct:
                if prediction_str in labels_correct:
                    labels_correct[prediction_str] = labels_correct[prediction_str] + 1
                else:
                    labels_correct[prediction_str] = 1

        prediction_accuracy = f"{model}\nCorrect: {count_correct}\nTotal:{total}\nAccuracy: {count_correct / total}\n"
        print(prediction_accuracy)

        prediction_evaluation_str = '\n'.join(prediction_evaluation)
        label_count_str = '\n'.join([' - '.join([key, str(value)]) for key, value in label_count.items()])
        labels_correct_str = '\n'.join([' - '.join([key, str(value)]) for key, value in labels_correct.items()])

        prediction_statistics = f"{prediction_accuracy}\nPredicted Label Counts:\n{label_count_str}\n\nCount True Label Correct:\n{labels_correct_str}\n\nTest Set Predictions:\n{prediction_evaluation_str}\n\n"

        return prediction_statistics

    def check_classification_accuracy(self, prediction, label):
        return prediction == label

    def check_regression_accuracy(self, prediction, label):
        checks = []

        for item in list(zip(prediction, label)):
            difference = abs(item[0] - item[1])
            if difference < 0.1:
                checks.append(True)
            else:
                checks.append(False)

        return all(checks)

    def write_output_to_file(self, statistics):
        for model, statistic in statistics.items():
            with open(os.path.join(self.config['files']['output'], f"random_forest_{model}_prediction_summary.txt"), "w") as file:
                file.write(statistic)

    def extract_info_dataset(self, group, val_lipids, train_set, test_set):
        if isinstance(group, h5py.Group):
            for key, value in group.items():
                self.extract_info_dataset(value, val_lipids, train_set, test_set)
        else:
            data_array = self.get_spectrum_data(group)

            if group.attrs['lipid_species'] in val_lipids:
                test_set.append(data_array)
            else:
                train_set.append(data_array)

    def extract_info_dataset_no_split(self, group, dataset):
        if isinstance(group, h5py.Group):
            for key, value in group.items():
                self.extract_info_dataset_no_split(value, dataset)
        else:
            data_array = self.get_spectrum_data(group)
            dataset.append(data_array)

    def extract_features_and_labels(self, dataset):
        dataset = shuffle(dataset, random_state=0)
        features = [row[0] for row in dataset]
        labels = [row[1:] for row in dataset]

        return features, labels

    def prepare_data(self):
        dataset_file = h5py.File(self.config['files']['train_input'], "r")

        if self.config['files']["splitting_instructions"]:
            if self.config['files']["splitting_instructions"] == 'leakage':
                dataset = []
                self.extract_info_dataset_no_split(dataset_file['/all_datasets/'], dataset)

                dataset_features, dataset_labels = self.extract_features_and_labels(dataset)
                train_features, test_features, train_labels, test_labels = train_test_split(dataset_features, dataset_labels)

            else:
                train_set = []
                test_set = []
                splitting_instructions = read_yaml(self.config['files']["splitting_instructions"])
                self.extract_info_dataset(dataset_file['/lipid_classes/'], splitting_instructions['val'], train_set, test_set)

                train_features, train_labels = self.extract_features_and_labels(train_set)
                test_features, test_labels = self.extract_features_and_labels(test_set)
        else:
            # TODO: implement random lipid species split
            train_features = None
            train_labels = None
            test_features = None
            test_labels = None

        dataset_file.close()

        return train_features, train_labels, test_features, test_labels


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    arguments = parser.parse_args()

    config = read_yaml(arguments.config)
    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%d/%m/%Y - %H:%M:%S")

    logging.info('Random forest run started.')

    config = get_config()

    random_forest = RandomForest(config)
    random_forest.run()

    logging.info('Random forest run completed.')
