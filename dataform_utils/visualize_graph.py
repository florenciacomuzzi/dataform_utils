"""
Visualize a DAG from the output of `dataform compile --json`
"""
import random

import matplotlib.colors as mcolors
from typing import List, Dict
import logging
import json

from graphlib import TopologicalSorter
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network


logger = logging.getLogger(__name__)


class Action:
    def __init__(self, data):
        self.file_name = data['fileName']
        self.typ = data['type']
        self.database = data['target']['database']
        self.schema = data['target']['schema']
        self.name = data['target']['name']
        self.target_name = self.build_target_name(data['target']['database'], data['target']['schema'],
                                                  data['target']['name'])
        self.tags = data.get('tags', None)
        self.description = data['actionDescriptor']['description'] if data.get('actionDescriptor') else None
        self.dependency_targets = [self.build_target_name(dep['database'], dep['schema'], dep['name'])
                                   for dep in data['dependencyTargets']] if data.get('dependencyTargets') else None
        self.parent_action = self.build_target_name(
            data['parentAction']['database'],data['parentAction']['schema'],
            data['parentAction']['name']) if data.get('parentAction') is not None else None

    @staticmethod
    def build_target_name(db, schema, name):
        return f"{db}.{schema}.{name}"

    def to_dict(self):
        return {
            'file_name': self.file_name,
            'type': self.typ,
            'database': self.database,
            'schema': self.schema,
            'name': self.name,
            'target_name': self.target_name,
            'tags': self.tags,
            'description': self.description,
            'dependency_targets': self.dependency_targets,
            'parent_action': self.parent_action
        }

    def __repr__(self):
        return f"Action({self.name})"


def load_data(path):
    data = None
    with open(path, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as e:
            logger.error(f"Error: {e}")
            return {}
    actions = {}
    # for item in [*data['tables'], *data['operations'], *data['assertions']]:
    for item in [*data['tables'], *data['operations']]:
        if not item.get('name'):
            item['name'] = Action.build_target_name(item['target']['database'], item['target']['schema'],
                                                    item['target']['name'])
        if not item.get('type'):
            item['type'] = "operation"
        actions[item['name']] = Action(item)
    return actions


class DynamicGraph:
    def __init__(self, actions):
        self.graph = TopologicalSorter()
        self.dependencies = {}  # Track dependencies for visualization
        self.actions = actions

    def add_node(self, node, dependencies=()):
        try:
            self.graph.add(node, *dependencies)
            for dep in dependencies:
                self.dependencies.setdefault(node, []).append(dep)
        except ValueError as e:
            print(f"Error adding node {node}: {e}")

    def add_dependency(self, node, dependency):
        self.add_node(node, (dependency,))

    def get_order(self):
        """
        Get a possible topological order of the added nodes.

        :return: A list of nodes in a topologically sorted order.
        """
        return list(self.graph.static_order())

    def visualize(self):
        # Create a directed graph
        G = nx.MultiDiGraph(directed=True)

        tag_color, used_colors = self.tag_to_color()

        # Add nodes and edges to the graph
        for node, deps in self.dependencies.items():
            attrs = self.actions[node].to_dict() if self.actions.get(node) else {}
            G.add_node(node, key=node, **attrs)
            node_colors = []
            if attrs['tags'] is not None:
                for tag in attrs['tags']:
                    node_colors.append(tag_color[tag])
            for dep in deps:
                dep_colors = []
                if self.actions.get(dep) and self.actions[dep].tags is not None:
                    for tag in self.actions[dep].tags:
                        dep_colors.append(tag_color[tag])
                edge_colors = ':'.join(node_colors + dep_colors)
                if edge_colors == "":
                    edge_colors = "pink"
                G.add_edge(dep, node, color=edge_colors)

        for layer, nodes in enumerate(nx.topological_generations(G)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                G.nodes[node]["layer"] = layer

        # Compute the multipartite_layout using the "layer" node attribute
        pos = nx.multipartite_layout(G, subset_key="layer")
        fig, ax = plt.subplots()
        edges = G.edges()
        nx.draw_networkx(G, pos=pos, ax=ax, node_size=5, node_color="white", font_size=4, font_weight="normal",
                         edge_color=used_colors)
        ax.set_title("DAG layout in topological order")
        fig.tight_layout()
        plt.show()

        # Plot with pyvis
        net = Network(
            directed=True,
            select_menu=True,  # Show part 1 in the plot (optional)
            filter_menu=True,  # Show part 2 in the plot (optional)
            height=800,
            width=800,
            neighborhood_highlight=True
        )
        net.repulsion()
        net.show_buttons(filter_=['nodes', 'edges', 'physics'])  # Show part 3 in the plot (optional)
        net.from_nx(G)  # Create directly from nx graph
        net.show('graph.html', notebook=False)

    def tag_to_color(self):
        tags = set()
        for item in self.actions.values():
            if item.tags is not None:
                for tag in item.tags:
                    tags.add(tag)

        used_colors = set()
        available_colors = set(mcolors.CSS4_COLORS.keys())
        tag_color_map = {}
        for tag in tags:
            color = random.choice(list(available_colors))
            tag_color_map[tag] = color
            used_colors.add(color)
            available_colors.remove(color)
        return tag_color_map, used_colors


if __name__ == '__main__':
    acts = load_data('graph.json')
    nodes = {}
    dynamic_graph = DynamicGraph(acts)
    dynamic_graph.add_node("root")
    for key, act in acts.items():
        if act.parent_action is not None:  # act.parent_action is a target_name
            dynamic_graph.add_dependency(act.target_name, act.parent_action)
        elif act.dependency_targets is not None:
            for dep in act.dependency_targets:
                dynamic_graph.add_dependency(act.target_name, dep)
    dynamic_graph.visualize()
