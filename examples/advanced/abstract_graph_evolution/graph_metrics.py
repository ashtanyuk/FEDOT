from datetime import timedelta
from typing import Optional, Callable, Dict

import netcomp
import networkx as nx
import numpy as np
from networkx import graph_edit_distance

from fedot.core.optimisers.gp_comp.pipeline_composer_requirements import PipelineComposerRequirements


def get_edit_dist_metric(target_graph: nx.DiGraph,
                         requirements: Optional[PipelineComposerRequirements] = None,
                         timeout=timedelta(seconds=60),
                         ) -> Callable[[nx.DiGraph], float]:

    def node_match(node_content_1: Dict, node_content_2: Dict) -> bool:
        operations_do_match = node_content_1.get('name') == node_content_2.get('name')
        return True or operations_do_match

    if requirements:
        upper_bound = int(np.sqrt(requirements.max_depth * requirements.max_arity)),
        timeout = timeout or requirements.max_pipeline_fit_time
    else:
        upper_bound = None

    def metric(graph: nx.DiGraph) -> float:
        ged = graph_edit_distance(target_graph, graph,
                                  node_match=node_match,
                                  upper_bound=upper_bound,
                                  timeout=timeout.seconds if timeout else None,
                                 )
        return ged or upper_bound

    return metric


def matrix_edit_dist(target_graph: nx.DiGraph, graph: nx.DiGraph) -> float:
    target_adj = nx.adjacency_matrix(target_graph)
    adj = nx.adjacency_matrix(graph)
    value = netcomp.edit_distance(target_adj, adj)
    return value


def spectral_dist(target_graph: nx.DiGraph, graph: nx.DiGraph, k: int = 10) -> float:
    target_adj = nx.adjacency_matrix(target_graph)
    adj = nx.adjacency_matrix(graph)
    num_eigenvalues = min(k, target_graph.number_of_nodes(), graph.number_of_nodes())
    value = netcomp.lambda_dist(target_adj, adj, kind='laplacian', k=num_eigenvalues)
    return value
