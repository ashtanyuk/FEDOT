from functools import partial
from typing import Union

import numpy as np

from fedot.api.api_utils.assumptions.assumptions_builder \
    import UniModalAssumptionsBuilder, MultiModalAssumptionsBuilder, AssumptionsBuilder
from fedot.api.api_utils.assumptions.preprocessing_builder import PreprocessingBuilder
from fedot.core.data.data import InputData
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.log import default_log
from fedot.core.pipelines.node import Node, PrimaryNode
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.operation_types_repository import OperationTypesRepository
from fedot.core.repository.tasks import TaskTypesEnum, Task, TsForecastingParams
from fedot.preprocessing.data_types import TableTypesCorrector
from fedot.preprocessing.preprocessing import DataPreprocessor
from test.unit.pipelines.test_pipeline_builder import pipelines_same

from test.unit.data_operations.test_data_operations_implementations \
    import get_time_series, get_small_regression_dataset
from test.unit.api.test_main_api \
    import get_dataset, load_categorical_unimodal
from test.unit.multimodal.data_generators import get_single_task_multimodal_tabular_data


def pipeline_contains_one(pipeline: Pipeline, operation_name: str) -> bool:
    def is_the_operation(node: Node):
        return node.operation.operation_type == operation_name

    return any(map(is_the_operation, pipeline.nodes))


def pipeline_contains_all(pipeline: Pipeline, *operation_name: str) -> bool:
    contains_one = partial(pipeline_contains_one, pipeline)
    return all(map(contains_one, operation_name))


def pipeline_contains_any(pipeline: Pipeline, *operation_name: str) -> bool:
    contains_one = partial(pipeline_contains_one, pipeline)
    return any(map(contains_one, operation_name))


def get_suitable_operations_for_task(task_type: TaskTypesEnum, data_type: DataTypesEnum, repo='model'):
    operations, _ = OperationTypesRepository(repo).suitable_operation(task_type=task_type, data_type=data_type)
    return operations


def get_test_ts_gaps_data():
    from examples.simple.time_series_forecasting.gapfilling import get_array_with_gaps

    gaps_array = get_array_with_gaps(gap_value=np.nan)
    data_ts_gaps = InputData(idx=np.arange(0, len(gaps_array)),
                             features=gaps_array,
                             target=gaps_array,
                             task=Task(TaskTypesEnum.ts_forecasting,
                                       TsForecastingParams(forecast_length=5)),
                             data_type=DataTypesEnum.ts)
    return data_ts_gaps


def preprocess(task_type: TaskTypesEnum, data: Union[InputData, MultiModalData]) -> Pipeline:
    return PreprocessingBuilder.builder_for_data(task_type, data).to_pipeline() or Pipeline()


def test_preprocessing_builder_no_data():
    assert pipeline_contains_all(PreprocessingBuilder(TaskTypesEnum.regression).to_pipeline(), 'scaling')
    assert pipeline_contains_all(PreprocessingBuilder(TaskTypesEnum.regression).with_gaps().to_pipeline(),
                                 'simple_imputation')
    assert pipeline_contains_all(PreprocessingBuilder(TaskTypesEnum.regression).with_categorical().to_pipeline(),
                                 'one_hot_encoding')
    assert pipeline_contains_all(
        PreprocessingBuilder(TaskTypesEnum.regression).with_gaps().with_categorical().to_pipeline(),
        'simple_imputation', 'one_hot_encoding')

    # have default preprocessing pipelines
    assert PreprocessingBuilder(TaskTypesEnum.regression).to_pipeline() is not None
    assert PreprocessingBuilder(TaskTypesEnum.classification).to_pipeline() is not None
    assert PreprocessingBuilder(TaskTypesEnum.clustering).to_pipeline() is not None

    # have no default preprocessing pipelines without additional options
    assert PreprocessingBuilder(TaskTypesEnum.ts_forecasting).to_pipeline() is None
    # with additional options ok
    assert PreprocessingBuilder(TaskTypesEnum.ts_forecasting).with_gaps().to_pipeline() is not None


def test_preprocessing_builder_with_data():
    # TableTypesCorrector fills in .supplementary_data needed for preprocessing_builder
    data_reg = TableTypesCorrector().convert_data_for_fit(get_small_regression_dataset()[0])
    data_cats = TableTypesCorrector().convert_data_for_fit(load_categorical_unimodal()[0])
    data_ts, _, _ = get_time_series()
    data_ts_gaps = get_test_ts_gaps_data()

    assert pipeline_contains_all(preprocess(TaskTypesEnum.regression, data_reg), 'scaling')
    assert pipeline_contains_all(preprocess(TaskTypesEnum.classification, data_cats), 'one_hot_encoding')

    assert not pipeline_contains_one(preprocess(TaskTypesEnum.ts_forecasting, data_ts), 'simple_imputation')
    assert not pipeline_contains_one(preprocess(TaskTypesEnum.ts_forecasting, data_ts), 'scaling')
    assert pipeline_contains_all(preprocess(TaskTypesEnum.ts_forecasting, data_ts_gaps), 'simple_imputation')


def test_assumptions_builder_for_multimodal_data():
    mm_data, _ = get_single_task_multimodal_tabular_data()
    logger = default_log('FEDOT logger', verbose_level=4)

    mm_builder = MultiModalAssumptionsBuilder(mm_data).with_logger(logger)
    mm_pipeline: Pipeline = mm_builder.build()[0]

    assert pipeline_contains_all(mm_pipeline, *mm_data)
    assert len(list(filter(lambda node: isinstance(node, PrimaryNode), mm_pipeline.nodes)))
    assert len(mm_pipeline.root_node.nodes_from) == mm_data.num_classes
    assert mm_pipeline.length == mm_pipeline.depth * mm_data.num_classes - 1  # minus final ensemble node


def test_assumptions_builder_unsuitable_available_operations():
    """ Check that when we provide unsuitable available operations then they're
    ignored, and we get same pipelines as without specifying available operations. """

    train_input, _, _ = get_dataset(task_type='classification')
    train_input = DataPreprocessor().obligatory_prepare_for_fit(train_input)
    logger = default_log('FEDOT logger', verbose_level=4)
    available_operations = ['linear', 'xgboost', 'lagged']

    default_builder = UniModalAssumptionsBuilder(train_input).with_logger(logger)
    checked_builder = UniModalAssumptionsBuilder(train_input).with_logger(logger) \
        .from_operations(available_operations)

    assert default_builder.build() == checked_builder.build()


def test_assumptions_builder_suitable_available_operations_unidata():
    """ Check that when we provide suitable available operations then we get fallback pipeline. """

    task = Task(TaskTypesEnum.classification)
    train_input, _, _ = get_dataset(task_type='classification')
    train_input = DataPreprocessor().obligatory_prepare_for_fit(train_input)

    impl_test_assumptions_builder_suitable_available_operations(task, train_input)


def impl_test_assumptions_builder_suitable_available_operations(
        task, train_input, data_type=None, logger=default_log('FEDOT logger', verbose_level=4)):
    """ Check that available operations, when they are suitable for the task,
    are taken into account by AssumptionsBuilder. This is implementation part of the test. """
    if not data_type:
        data_type = train_input.data_type
    available_operations = get_suitable_operations_for_task(task.task_type, data_type)
    assert available_operations

    default_builder = AssumptionsBuilder.get(train_input).with_logger(logger)
    baseline_pipeline = default_builder.build()[0]
    baseline_operation = baseline_pipeline.root_node.operation.operation_type
    available_operations.remove(baseline_operation)

    checked_builder = AssumptionsBuilder.get(train_input).with_logger(logger) \
        .from_operations(available_operations)
    checked_pipeline = checked_builder.build()[0]

    # check that results differ
    assert not pipelines_same(baseline_pipeline, checked_pipeline)
    # check expected structure of pipeline
    assert pipeline_contains_any(checked_pipeline, *available_operations)
