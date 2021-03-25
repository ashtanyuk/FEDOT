import warnings

from typing import Optional

from fedot.core.operations.evaluation.operation_implementations.models.\
    discriminant_analysis import LDAImplementation, QDAImplementation

from fedot.core.operations.evaluation.operation_implementations.\
    data_operations.sklearn_selectors import LinearClassFS, NonLinearClassFS

from fedot.core.data.data import InputData, OutputData
from fedot.core.operations.evaluation.evaluation_interfaces import EvaluationStrategy, SkLearnEvaluationStrategy
from fedot.core.repository.dataset_types import DataTypesEnum

warnings.filterwarnings("ignore", category=UserWarning)


class SkLearnClassificationStrategy(SkLearnEvaluationStrategy):
    """ Strategy for applying classification algorithms from Sklearn library """

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Predict method for classification task

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return: prediction target
        """
        n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(predict_data.features)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(predict_data.features)
            if n_classes < 2:
                raise NotImplementedError()
            elif n_classes == 2 and self.output_mode != 'full_probs':
                prediction = prediction[:, 1]
        else:
            raise ValueError(f'Output model {self.output_mode} is not supported')

        converted = OutputData(idx=predict_data.idx,
                               features=predict_data.features,
                               predict=prediction,
                               task=predict_data.task,
                               target=predict_data.target,
                               data_type=DataTypesEnum.table)
        return converted


class CustomClassificationStrategy(EvaluationStrategy):

    __operations_by_types = {
        'lda': LDAImplementation,
        'qda': QDAImplementation
    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        self.operation_impl = self._convert_to_operation(operation_type)
        super().__init__(operation_type, params)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained data operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            operation_implementation = self.operation_impl(**self.params_for_fit)
        else:
            operation_implementation = self.operation_impl()

        operation_implementation.fit(train_data.features, train_data.target)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Predict method for classification task

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return: prediction target
        """
        n_classes = len(trained_operation.classes_)
        if self.output_mode == 'labels':
            prediction = trained_operation.predict(predict_data.features)
        elif self.output_mode in ['probs', 'full_probs', 'default']:
            prediction = trained_operation.predict_proba(predict_data.features)
            if n_classes < 2:
                raise NotImplementedError()
            elif n_classes == 2 and self.output_mode != 'full_probs':
                prediction = prediction[:, 1]
        else:
            raise ValueError(
                f'Output model {self.output_mode} is not supported')

        converted = OutputData(idx=predict_data.idx,
                               features=predict_data.features,
                               predict=prediction,
                               task=predict_data.task,
                               target=predict_data.target,
                               data_type=DataTypesEnum.table)
        return converted

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain Custom Classification Strategy for {operation_type}')

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))


class CustomClassificationPreprocessingStrategy(EvaluationStrategy):
    """ Strategy for applying custom algorithms from FEDOT to preprocess data
    for classification task
    """

    __operations_by_types = {
        'rfe_lin_class': LinearClassFS,
        'rfe_non_lin_class': NonLinearClassFS
    }

    def __init__(self, operation_type: str, params: Optional[dict] = None):
        super().__init__(operation_type, params)
        self.operation_impl = self._convert_to_operation(operation_type)

    def fit(self, train_data: InputData):
        """
        This method is used for operation training with the data provided
        :param InputData train_data: data used for operation training
        :return: trained data operation
        """

        warnings.filterwarnings("ignore", category=RuntimeWarning)
        if self.params_for_fit:
            operation_implementation = self.operation_impl(**self.params_for_fit)
        else:
            operation_implementation = self.operation_impl()

        operation_implementation.fit(train_data.features)
        return operation_implementation

    def predict(self, trained_operation, predict_data: InputData,
                is_fit_chain_stage: bool):
        """
        Transform data

        :param trained_operation: model object
        :param predict_data: data used for prediction
        :param is_fit_chain_stage: is this fit or predict stage for chain
        :return:
        """
        # Prediction here is already OutputData type object
        prediction = trained_operation.transform(predict_data,
                                                 is_fit_chain_stage)
        return prediction

    def _convert_to_operation(self, operation_type: str):
        if operation_type in self.__operations_by_types.keys():
            return self.__operations_by_types[operation_type]
        else:
            raise ValueError(f'Impossible to obtain custom classification preprocessing strategy for {operation_type}')

    @property
    def implementation_info(self) -> str:
        return str(self._convert_to_operation(self.operation_type))
