from typing import Optional

from log_calls import record_history
import networkx as nx
from networkx.algorithms.cycles import simple_cycles
from networkx.algorithms.isolate import isolates

from fedot.core.chains.chain import Chain
from fedot.core.chains.chain_convert import chain_as_nx_graph
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.repository.tasks import Task

ERROR_PREFIX = 'Invalid chain configuration:'


@record_history(enabled=False)
def validate(chain: Chain, task: Optional[Task] = None):
    # TODO pass task to this function
    has_one_root(chain)
    has_no_cycle(chain)
    has_no_self_cycled_nodes(chain)
    has_no_isolated_nodes(chain)
    has_primary_nodes(chain)
    has_correct_model_positions(chain, task)
    return True


@record_history(enabled=False)
def has_one_root(chain: Chain):
    if chain.root_node:
        return True


@record_history(enabled=False)
def has_no_cycle(chain: Chain):
    graph, _ = chain_as_nx_graph(chain)
    cycled = list(simple_cycles(graph))
    if len(cycled) > 0:
        raise ValueError(f'{ERROR_PREFIX} Chain has cycles')

    return True


@record_history(enabled=False)
def has_no_isolated_nodes(chain: Chain):
    graph, _ = chain_as_nx_graph(chain)
    isolated = list(isolates(graph))
    if len(isolated) > 0 and chain.length != 1:
        raise ValueError(f'{ERROR_PREFIX} Chain has isolated nodes')
    return True


@record_history(enabled=False)
def has_primary_nodes(chain: Chain):
    if not any(node for node in chain.nodes if isinstance(node, PrimaryNode)):
        raise ValueError(f'{ERROR_PREFIX} Chain does not have primary nodes')
    return True


@record_history(enabled=False)
def has_no_self_cycled_nodes(chain: Chain):
    if any([node for node in chain.nodes if isinstance(node, SecondaryNode) and node in node.nodes_from]):
        raise ValueError(f'{ERROR_PREFIX} Chain has self-cycled nodes')
    return True


@record_history(enabled=False)
def has_no_isolated_components(chain: Chain):
    graph, _ = chain_as_nx_graph(chain)
    ud_graph = nx.Graph()
    ud_graph.add_nodes_from(graph)
    ud_graph.add_edges_from(graph.edges)
    if not nx.is_connected(ud_graph):
        raise ValueError(f'{ERROR_PREFIX} Chain has isolated components')
    return True


@record_history(enabled=False)
def _is_data_merged(chain: Chain):
    root_node_merges_data = 'composition' in chain.root_node.model_tags
    merging_is_required = any('decomposition' in node.model_tags for node in chain.nodes)
    data_merged_or_merging_not_required = root_node_merges_data or not merging_is_required

    return data_merged_or_merging_not_required


@record_history(enabled=False)
def _is_root_not_datamodel(chain: Chain):
    return 'data_model' not in chain.root_node.model_tags and \
           'decomposition' not in chain.root_node.model_tags


@record_history(enabled=False)
def has_correct_model_positions(chain: Chain, task: Optional[Task] = None):
    is_root_satisfy_task_type = True
    if task:
        is_root_satisfy_task_type = task.task_type in chain.root_node.model.acceptable_task_types

    if not _is_root_not_datamodel(chain) or \
            not _is_data_merged(chain) or \
            not is_root_satisfy_task_type:
        raise ValueError(f'{ERROR_PREFIX} Chain has incorrect models positions')

    return True
