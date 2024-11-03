import time, random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Network():
    def __init__(self, topology:int = 0, num_flows:int = 10, lambda_min:int = 10, lambda_max:int = 300):
        """
        topology= 0, 1, 2
        """
        self.topology = topology
        self.num_flows = num_flows

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        
        match self.topology:
            case 0:
                self.Graph = nx.grid_2d_graph(5, 5)
                self.Graph = nx.convert_node_labels_to_integers(self.Graph) # Convert Nodes into integers
            case 1:
                ...
            case 2:
                ...
    
    def update_weights(self):
        """ 
        Randomly updates weights on existing edges
        """
        for _ in range(self.num_flows):
            nodes = list(self.Graph.nodes())
            src, dst = random.sample(nodes, 2)  # Randomly choose two nodes
            arrival_rate = np.random.uniform(self.lambda_min, self.lambda_max)  # Random arrival rate

            increase_or_decrease = random.sample([-1, 1], 1)[0] # Randomly choose to increase of decrease

            if self.Graph.has_edge(src, dst):
                if 'weight' in self.Graph[src][dst]:
                    self.Graph[src][dst]['weight'] += (increase_or_decrease * arrival_rate)  # Increment existing weight
                else:
                    self.Graph[src][dst]['weight'] = arrival_rate  # Initialize weight if not present
    
    def plot_graph(self):
        """
        Plot the graph with current weights
        """
        pos = nx.spring_layout(self.Graph)
        nx.draw_networkx_nodes(self.Graph, pos, node_size=500, node_color='skyblue')
        
        nx.draw_networkx_edges(self.Graph, pos, width=1.5, alpha=0.7) # Draw edges
        edge_labels = nx.get_edge_attributes(self.Graph, 'weight')  # Get edge weights
        nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8) # Draw edge weights

        nx.draw_networkx_labels(self.Graph, pos, font_size=10, font_family="sans-serif") # Draw node labels

        plt.title(f"Network Topology {self.topology}")
        plt.axis('off')
        plt.show()

    def run(self, interval:int = 1, is_plot:bool = False):
        """
        Main function to run the network updates and optionally plot the graph
        """
        if is_plot:
            fig, ax = plt.subplots()
            pos = nx.spring_layout(self.Graph)

            def update(frame):
                self.update_weights()
                ax.clear()

                nx.draw_networkx_nodes(self.Graph, pos, node_size=500, node_color='skyblue', ax=ax)
                nx.draw_networkx_edges(self.Graph, pos, width=1.5, alpha=0.7, ax=ax)
                
                edge_labels = nx.get_edge_attributes(self.Graph, 'weight')
                edge_labels = {edge: f"{weight:.2f}" for edge, weight in edge_labels.items()}  # Format weights
                nx.draw_networkx_edge_labels(self.Graph, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8, ax=ax)
                
                nx.draw_networkx_labels(self.Graph, pos, font_size=10, font_family="sans-serif", ax=ax)
                ax.set_title(f"Network Topology {self.topology}")
                ax.axis('off')
            
            ani = FuncAnimation(fig, update, interval=interval * 1000)
            plt.show()
        else:
            while True:
                self.update_weights()
                time.sleep(interval)

if __name__ == '__main__':
    network = Network()
    network.run(interval=1, is_plot=True)
    # network.plot_graph()
