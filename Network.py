import time, random

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class Network():
    def __init__(self, topology:int = 0, service_rate:int = 3000, system_capacity:int = 10000):
        """
        topology= 0, 1, 2
        """
        self.topology = topology
        self.t = 0 # Tracks t

        match self.topology:
            case 0:
                self.num_of_switches = 25 # N = number of switches
                self.Graph = nx.grid_2d_graph(5, 5)
                self.Graph = nx.convert_node_labels_to_integers(self.Graph) # Convert Nodes into integers
            case 1:
                self.num_of_switches = 40 # N = number of switches
                self.Graph = nx.gnm_random_graph(self.num_of_switches, 61)
                while not nx.is_connected(self.Graph):
                    self.Graph = nx.gnm_random_graph(40, 60)

                pos = nx.spring_layout(self.Graph, seed=42)
                nx.draw(self.Graph, pos, with_labels=True, node_color='orange', node_size=500, font_size=8, font_color='black')
                plt.title("InternetMCI Topology with 19 Switches")
                plt.show()
            case 2:
                self.num_of_switches = 19 # N = number of switches
                self.Graph = nx.gnm_random_graph(self.num_of_switches, 33)
                while not nx.is_connected(self.Graph):
                    self.Graph = nx.gnm_random_graph(self.num_of_switches, 33)

                # self.Graph = nx.Graph()

                # for i in range(self.num_of_switches):
                #     self.Graph.add_node(i)
                
                # backbone_nodes = list(range(5)) # 5 Backbone Nodes
                # regional_nodes = list(set(range(self.num_of_switches)) - set(backbone_nodes)) # Regional Nodes

                # for i in range(len(backbone_nodes)):
                #     for j in range(i + 1, len(backbone_nodes)):
                #         self.Graph.add_edge(backbone_nodes[i], backbone_nodes[j])
                
                # for node in regional_nodes:
                #     self.Graph.add_edge(node, backbone_nodes[node % len(backbone_nodes)])


        # TODO: Initialize for t=0, and populate at runtime.
        self.M = 5 # Number of flows for a given time ; TODO: Needs a proper value and changes for every t.
        nodes = list(self.Graph.nodes)
        self.f = np.array([(random.choice(nodes), random.choice(nodes)) for _ in range(self.M)], dtype='i,i') # List of (src, dst) tuples of size self.M
        
        self.lam = np.random.uniform(10, 300, self.M) # Aggregate arrival 
        self.u = np.full(self.num_of_switches, service_rate) # Servivce rate of each switch ; Constant rate, for later work we can vary rates.
        self.K = np.full(self.num_of_switches, system_capacity) # Total system capacity of each switch. ; Constant system capacity, for later work we can vary rates.

        # Aggregate arrival rate per switch
        switch_lam = np.zeros(self.num_of_switches) # Need to keep track of this for all switches.
        for src, dst in self.f:
            switch_lam[src] += random.choice(self.lam)  # Assign a flow's rate to the source switch

        self.rho = switch_lam / self.u # Traffic Intensity for each switch.
        
        self.e_n = self.get_queue_occupation() # Returns an array of expected queue occupation of each switch.  
        self.P = self.get_P() # Returns a probability array(of packets getting lost) of size self.num_of_switches.
        self.e_d = self.get_expected_delays() # Returns an expected delay for each switch.

        self.d_k_e2e = self.get_end_end_delay() # Returns an array of size self.M(for each active flow at time t) ; self.d_k_e2e.mean() would be the e2e mean of the network.

        self.e_l = self.get_expected_loss() # Returns an array of expected loss of size self.num_of_switches.
    
    def shortest_path(self, src, dest):
        """
        Gets shortest path given src, dest in the graph.
        """
        return nx.shortest_path(G, source=src, target=dst, weight= lambda s,d,a: self.Graph[s][d]['weight'], method='dijkstra')
    
    def get_queue_occupation(self):
        """
        Gets queue occupation of each switch.
        """
        N = np.zeros(len(self.rho))
        for i in range(len(self.rho)):
            if r[i] < 1:
                N[i] = (r[i] / (1 - r[i])) - (((self.K[i] + 1) * (r ** (self.K[i] + 1))) / ((1 - r[i]) ** (self.K[i] + 1)))
            if r[i] == 1:
                N[i] = self.K[i] / 2
        return N

    def get_P(self):
        """
        Gets probability array of packets getting lost of size self.num_of_switches.
        """
        return ( (1 - self.rho) * (self.rho ** self.K) ) / ( 1 - (self.rho ** (self.K + 1)) )
    
    def get_expected_delays(self):
        """
        Gets expected delay for each switch.
        """
        return self.e_n / (self.lam * (1 - self.P))
    
    def get_end_end_delay(self):
        """
        Gets end to end delay for each flow in time self.t.
        """
        D = np.zeros(len(self.f))
        for i in range(len(self.f)):
            flow = self.f[i]
            p_star = self.shortest_path(flow[0], flow[1])
            for n in p_star:
                D[i] += self.e_d[n]
        return D
    
    def get_expected_loss(self):
        """
        Gets expected loss for each switch.
        """
        return self.lam * self.P
    
    def update_weights(self, weights):
        """
        Updates the weights of existing edges using the provided array of weights.

        Args:
            weights (list or np.array): An array of weights with a length equal to the number of edges.
        """
        # Check if the length of weights matches the number of edges
        if len(weights) != len(self.Graph.edges):
            raise ValueError("Length of weights array must match the number of edges in the graph.")

        # Iterate through the edges and update their weights
        for (edge, weight) in zip(self.Graph.edges, weights):
            self.Graph[edge[0]][edge[1]]['weight'] = weight

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

    # TODO: Needs configuring/Changing
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
    # network.run(interval=1, is_plot=True)
    # network.plot_graph()
