import random

import numpy as np

from fedot.core.dag.graph_delegate import GraphDelegate
from fedot.core.dag.linked_graph_node import LinkedGraphNode
from fedot.core.dag.verification_rules import has_no_self_cycled_nodes
from fedot.core.adapter import DirectAdapter
from fedot.core.optimisers.gp_comp.gp_optimizer import EvoGraphOptimizer
from fedot.core.optimisers.gp_comp.gp_params import GPGraphOptimizerParameters
from fedot.core.optimisers.gp_comp.operators.inheritance import GeneticSchemeTypesEnum
from fedot.core.optimisers.gp_comp.operators.mutation import MutationTypesEnum
from fedot.core.optimisers.gp_comp.operators.regularization import RegularizationTypesEnum
from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements
from fedot.core.optimisers.initial_graphs_generator import InitialPopulationGenerator
from fedot.core.optimisers.objective.objective import Objective
from fedot.core.optimisers.objective.objective_eval import ObjectiveEvaluate
from fedot.core.optimisers.optimizer import GraphGenerationParams
from fedot.core.dag.convert import graph_structure_as_nx_graph
from fedot.core.pipelines.pipeline_node_factory import PipelineOptNodeFactory

random.seed(1)
np.random.seed(1)


class CustomModel(GraphDelegate):

    def evaluate(self):
        return 0


class CustomNode(LinkedGraphNode):
    def __str__(self):
        return f'custom_{str(self.content["name"])}'


def custom_metric(custom_model: CustomModel):
    _, labels = graph_structure_as_nx_graph(custom_model)

    return -len(labels) + custom_model.evaluate()


def test_custom_graph_opt():
    """Test checks for the use case of custom graph optimisation:
    that it can be initialised without problem and returns sane result."""

    nodes_types = ['A', 'B', 'C', 'D']
    rules = [has_no_self_cycled_nodes]

    requirements = PipelineComposerRequirements(
        primary=nodes_types,
        secondary=nodes_types,
        num_of_generations=5,
        show_progress=False)

    optimiser_parameters = GPGraphOptimizerParameters(
        pop_size=5,
        genetic_scheme_type=GeneticSchemeTypesEnum.steady_state,
        mutation_types=[
            MutationTypesEnum.simple,
            MutationTypesEnum.reduce,
            MutationTypesEnum.growth,
            MutationTypesEnum.local_growth],
        regularization_type=RegularizationTypesEnum.none)

    graph_generation_params = GraphGenerationParams(
        adapter=DirectAdapter(CustomModel, CustomNode),
        rules_for_constraint=rules,
        node_factory=PipelineOptNodeFactory(requirements=requirements))

    objective = Objective({'custom': custom_metric})
    init_population = InitialPopulationGenerator(optimiser_parameters.pop_size,
                                                 graph_generation_params, requirements)()
    optimiser = EvoGraphOptimizer(
        graph_generation_params=graph_generation_params,
        objective=objective,
        graph_optimizer_params=optimiser_parameters,
        requirements=requirements,
        initial_graphs=init_population)

    objective_eval = ObjectiveEvaluate(objective)
    optimized_graphs = optimiser.optimise(objective_eval)
    optimized_network = optimiser.graph_generation_params.adapter.restore(optimized_graphs[0])

    assert optimized_network is not None
    assert isinstance(optimized_network, CustomModel)
    assert isinstance(optimized_network.nodes[0], CustomNode)
    assert optimized_network.length > 1
    assert optimized_network.depth > 1
