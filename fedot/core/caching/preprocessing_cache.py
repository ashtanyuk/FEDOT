from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from fedot.core.caching.base_cache import BaseCache
from fedot.core.caching.preprocessing_cache_db import PreprocessingCacheDB
from fedot.core.data.data import InputData, data_type_is_table
from fedot.core.data.multi_modal import MultiModalData
from fedot.core.operations.evaluation.operation_implementations.data_operations.categorical_encoders import (
    OneHotEncodingImplementation
)
from fedot.core.operations.evaluation.operation_implementations.data_operations.sklearn_transformations import (
    ImputationImplementation
)

if TYPE_CHECKING:
    from fedot.core.pipelines.pipeline import Pipeline


class PreprocessingCache(BaseCache):
    """
    Usage of that class makes sense only if you have table data.
    Stores/loads `DataPreprocessor`'s encoders and imputers for pipelines to decrease optional preprocessing time.

    :param db_path: optional str determining a file name for caching preprocessors items.
    """

    def __init__(self, db_path: Optional[str] = None):
        super().__init__(PreprocessingCacheDB(db_path))

    def try_load_preprocessor(self, pipeline: 'Pipeline', fold_id: Union[int, None]):
        """
        Tries to find preprocessor in DB table and load it for pipeline

        :param pipeline: pipeline to load preprocessor for
        :param fold_id: number of fold
        """
        try:
            structural_id = _get_db_uid(pipeline, fold_id)
            processors = self._db.get_preprocessor(structural_id)
            if processors:
                pipeline.encoder, pipeline.imputer = processors
        except Exception as exc:
            self.log.warning(f'Preprocessor search error: {exc}')

    def add_preprocessor(self, pipeline: 'Pipeline', fold_id: Optional[Union[int, None]] = None):
        """
        Adds preprocessor into DB working table.

        :param pipeline: pipeline with preprocessor to add
        :param fold_id: number of fold
        """
        structural_id = _get_db_uid(pipeline, fold_id)
        self._db.add_preprocessor(structural_id, pipeline.preprocessor)


def _get_db_uid(pipeline: 'Pipeline', fold_id: Union[int, None]) -> str:
    """
    Constructs unique id from pipeline and data, which is considered as primary key for DB.

    :param pipeline: pipeline to get uid from
    :param fold_id: number of fold

    :return: unique pipeline plus related data identificator
    """
    fold_id = fold_id if fold_id is not None else ""
    pipeline_id = pipeline.root_node.descriptive_id
    return f'{pipeline_id}_{fold_id}'
