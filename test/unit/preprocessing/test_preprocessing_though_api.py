from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from fedot.api.main import Fedot
from fedot.core.data.data import InputData
from fedot.core.data.data_split import train_test_data_setup
from fedot.core.data.supplementary_data import SupplementaryData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.utils import fedot_project_root
from fedot.preprocessing.data_types import NAME_CLASS_STR


def data_with_only_categorical_features():
    """ Generate tabular data with only categorical features. All of them are binary. """
    supp_data = SupplementaryData(column_types={'features': [NAME_CLASS_STR] * 3})
    task = Task(TaskTypesEnum.regression)
    features = np.array([["'a'", "0", "1"],
                         ["'b'", "1", "0"],
                         ["'c'", "1", "0"]], dtype=object)
    input_data = InputData(idx=[0, 1, 2], features=features,
                           target=np.array([0, 1, 2]),
                           task=task, data_type=DataTypesEnum.table,
                           supplementary_data=supp_data)

    return input_data


def data_with_categorical_features_and_nans():
    """ Generate tabular data with only categorical features. All of them are binary. """
    supp_data = SupplementaryData(column_types={'features': [NAME_CLASS_STR] * 3})
    task = Task(TaskTypesEnum.regression)
    features = np.array([["'a'", "0", "1"],
                         ['nan', "1", "0"],
                         ["'c'", "1", "0"],
                         ["'a'", "0", "1"],
                         ['nan', "1", "0"],
                         ["'c'", "1", "0"],
                         ["'a'", "0", "1"],
                         ["'c'", "1", "0"],
                         ["'c'", "1", "0"]])
    input_data = InputData(idx=[0, 1, 2, 3, 4, 5, 6, 7, 8], features=features,
                           target=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]),
                           task=task, data_type=DataTypesEnum.table,
                           supplementary_data=supp_data)

    return input_data


def data_with_too_much_nans():
    """ Generate tabular data with too much nan's in numpy array (inf values also must be signed as nan).
    Columns with ids 1 and 2 have nans more than 90% in their structure.
    """
    task = Task(TaskTypesEnum.regression)
    features = np.array([[1, np.inf, 8],
                         [2, np.inf, 10],
                         [3, np.inf, 11],
                         [14, np.inf, 11],
                         [13, np.inf, 11],
                         [63, np.inf, 11],
                         ['nan', np.inf, 11],
                         [81, np.inf, 11],
                         [7, np.inf, 14],
                         [8, np.nan, 15],
                         ['nan', np.nan, 23],
                         [19, np.inf, 6],
                         [29, np.inf, 4],
                         [39, np.inf, np.nan],
                         [49, '1', np.inf],
                         [99, '1', np.inf],
                         [8, np.nan, np.inf]])
    target = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10],
                       [11], [12], [13], [14], [15], [16]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_spaces_and_nans_in_features():
    """
    Generate InputData with categorical features with leading and
    trailing spaces. Dataset contains np.nan also.
    """
    task = Task(TaskTypesEnum.regression)
    features = np.array([['1 ', '1 '],
                         [np.nan, ' 0'],
                         [' 1 ', np.nan],
                         ['1 ', '0  '],
                         ['0  ', '  1'],
                         ['1 ', '  0']], dtype=object)
    target = np.array([[0], [1], [2], [3], [4], [5]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_nans_in_target_column():
    """ Generate InputData with np.nan values in target column """
    task = Task(TaskTypesEnum.regression)
    features = np.array([[1, 2],
                         [2, 2],
                         [0, 3],
                         [2, 3],
                         [3, 4],
                         [1, 3]])
    target = np.array([[0], [1], [np.nan], [np.nan], [4], [5]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_nans_in_multi_target():
    """
    Generate InputData with np.nan values in target columns.
    So the multi-output regression task is solved.
    """
    task = Task(TaskTypesEnum.regression)
    features = np.array([[1, 2],
                         [2, 2],
                         [0, 3],
                         [2, 3],
                         [3, 4],
                         [1, 3]])
    target = np.array([[0, 2], [1, 3], [np.nan, np.nan], [3, np.nan], [4, 4], [5, 6]])
    train_input = InputData(idx=[0, 1, 2, 3, 4, 5], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_categorical_target(with_nan: bool = False):
    """
    Generate dataset for classification task where target column is defined as
    string categories (e.g. 'red', 'green'). Dataset is generated so that when
    split into training and test in the test sample in the target will always
    be a new category.

    :param with_nan: is there a need to generate target column with np.nan
    """
    task = Task(TaskTypesEnum.classification)
    features = np.array([[0, 0],
                         [0, 1],
                         [8, 8],
                         [8, 9]])
    if with_nan:
        target = np.array(['blue', np.nan, np.nan, 'di'], dtype=object)
    else:
        target = np.array(['blue', 'da', 'ba', 'di'], dtype=str)
    train_input = InputData(idx=[0, 1, 2, 3], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_text_features():
    """ Generate tabular data with text features """
    task = Task(TaskTypesEnum.classification)
    features = np.array(['My mistress eyes are nothing like the sun.',
                         'Coral is far more red than her lips red.',
                         'If snow be white, why then her breasts are dun?',
                         'If hairs be wires, black wires grow on her head.'],
                        dtype=object)

    target = np.array([[0], [1], [0], [1]])
    train_input = InputData(idx=[0, 1, 2, 3], features=features,
                            target=target, task=task, data_type=DataTypesEnum.text,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_pseudo_text_features():
    """ Generate tabular data with text features """
    task = Task(TaskTypesEnum.classification)
    features = np.array([np.nan,
                         np.nan,
                         '4.2',
                         '3',
                         '1e-3',
                         np.nan],
                        dtype=object)

    target = np.array([[0], [1], [0], [1], [0]])
    train_input = InputData(idx=[0, 1, 2, 3, 4], features=features,
                            target=target, task=task, data_type=DataTypesEnum.table,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


def data_with_text_features_and_nans():
    """ Generate tabular data with text features """
    task = Task(TaskTypesEnum.classification)
    features = np.array([np.nan,
                         '',
                         'that can be stated,',
                         'is not the eternal',
                         np.nan],
                        dtype=object)

    target = np.array([[0], [1], [0], [1], [0]])
    train_input = InputData(idx=[0, 1, 2, 3, 4], features=features,
                            target=target, task=task, data_type=DataTypesEnum.text,
                            supplementary_data=SupplementaryData(was_preprocessed=False))

    return train_input


# TODO: @andreygetmanov (test data with image features)

def real_regression_data():
    data_frame = pd.read_csv(Path(fedot_project_root(), 'test/data/used_car.csv'),
                             sep=';')

    train_data, test_data = train_test_split(data_frame.values,
                                             test_size=0.1,
                                             random_state=42)
    train_data, test_data = pd.DataFrame(train_data), pd.DataFrame(test_data)
    return train_data, test_data


def test_correct_api_dataset_preprocessing():
    """ Check if dataset preprocessing was performed correctly when API launch using. """
    funcs = [real_regression_data]

    # Check for all datasets
    for data_generator in funcs:
        data = data_generator()
        if isinstance(data, tuple):
            train_data, test_data = data
        else:
            train_data, test_data = data, data
        fedot_model = Fedot(problem='regression')
        pipeline = fedot_model.fit(train_data, predefined_model='auto')
        predict = fedot_model.predict(test_data)
        assert pipeline is not None
        assert predict is not None


def test_categorical_target_processed_correctly():
    """ Check if categorical target for classification task first converted
    into integer values and then perform inverse operation. API tested in this
    test.
    """
    classification_data = data_with_categorical_target()
    train_data, test_data = train_test_data_setup(classification_data)

    fedot_model = Fedot(problem='classification')
    fedot_model.fit(train_data, predefined_model='auto')
    predicted = fedot_model.predict(test_data)

    # Predicted label must be close to 'di' label (so, right prediction is 'ba')
    assert predicted[0] == 'ba'


def test_correct_api_dataset_with_text_preprocessing():
    """ Check if dataset with text features preprocessing was performed correctly when API launch using. """
    funcs = [data_with_text_features, data_with_text_features_and_nans]

    # Check for all datasets
    for data_generator in funcs:
        input_data = data_generator()
        fedot_model = Fedot(problem='classification')
        fedot_model.fit(input_data, predefined_model='auto')
        predicted = fedot_model.predict(input_data)

        # Check the features were transformed during preprocessing
        assert fedot_model.prediction.features.shape[1] > 1
        assert fedot_model.prediction.features.shape[0] == input_data.features.shape[0]

        # Check if there is a text node in pipeline
        node_tags = [node.tags for node in fedot_model.current_pipeline.nodes]
        assert any('text' in current_tags for current_tags in node_tags)
        assert len(predicted) > 0


def test_correct_api_dataset_with_pseudo_text_preprocessing():
    """ Check if dataset with pseudo text features was preprocessed correctly (as numerical) when API launch using. """

    input_data = data_with_pseudo_text_features()
    fedot_model = Fedot(problem='classification')
    fedot_model.fit(input_data, predefined_model='auto')
    predicted = fedot_model.predict(input_data)

    # Check there are no text nodes in the pipeline
    node_tags = [node.tags for node in fedot_model.current_pipeline.nodes]
    assert not any('text' in current_tags for current_tags in node_tags)
    assert fedot_model.prediction.features.shape[0] == input_data.features.shape[0]
