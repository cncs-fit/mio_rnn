# helper functions for calculating modularity and other metrics

import networkx as nx
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def div_ev(div_list):
    '''
    Calculate the separation index based on the provided division list.
    Args:
        div_list: A list containing two arrays, each of size (2,N)
    '''

    a_d = np.abs(div_list[0]) - np.abs(div_list[1])
    sum_aw = np.sum(np.abs(div_list))

    g1_w = np.sum(a_d[:50])
    g2_w = np.sum(a_d[50:100])
    
    return abs((g1_w - g2_w) / sum_aw)


def calM(model, batch, node1, node2, batch_size=50):
    '''Calculate the modularity of the graph based on the model's hidden layer output.
    Args:
        model: The trained model.
        batch: Input batch data.
        node1: First node for modularity calculation.
        node2: Second node for modularity calculation.
        batch_size: Size of the batch.
    Returns:
        Modularity value of the graph.
    '''

    G = nx.Graph()

    sg = [] # list to store correlation coefficients
    mo = model.hidden_layer(batch) # get hidden layer output
    mo = tf.transpose(mo, perm=[0, 2, 1]) # transpose to shape (batch_size, N=100, timestep)
    mo = mo.numpy()

    for idx in range(batch_size):
        sg_t = np.corrcoef(mo[idx]) # N x N correlation matrix
        sg_t = np.abs(sg_t)
        sg.append(sg_t)
    sg = np.mean(sg, axis=0)  # average over the batch, resulting N x N matrix
    for i in range(1,100 + 1):
        G.add_nodes_from([str(i)])

    for j in range(100):
        for k in range(j+1,100):
            G.add_weighted_edges_from([(str(j+1), str(k+1), sg[j,k])])

    # print(nx.community.modularity(G, communities=[node1,node2]))
    return nx.community.modularity(G, communities=[node1,node2]) # type:ignore

def calD(model):
    '''Calculate the separation index of the model's output.
    Args:
        model: The trained model.
    Returns:
        D_out: Separation index value.
    '''

    l_1 = model.layers[1]
    omomi = np.array(l_1.get_weights()[0]) # get output weights (N,6)

    omomi = np.abs(omomi)
    omomi = omomi.transpose() # (6,N)

    o_ave_1 = (omomi[0,:] + omomi[1,:] + omomi[2,:]) / 3 #(N,)
    o_ave_2 = (omomi[3,:] + omomi[4,:] + omomi[5,:]) / 3 # (N,)

    temp_list_o = np.array([o_ave_1, o_ave_2]) # (2,N)
    return div_ev(temp_list_o)


def calS(model, batch, batch_size=50):
    '''Calculate the separation index of the model's hidden layer output.
    Args:
        model: The trained model.
        batch: Input batch data.
        batch_size: Size of the batch.
    Returns:
        D_cor: Separation index value.
    '''

    sl = []
    oo = model(batch)
    oo = tf.transpose(oo, perm=[0, 2, 1])
    mo = model.hidden_layer(batch) # 
    mo = tf.transpose(mo, perm=[0, 2, 1])
    mo = mo.numpy()
    for idx in range(batch_size):
        s = np.zeros((6,100))

        for t_2 in range(6):
            for t_1 in range(100):
                temp = np.corrcoef(mo[idx,t_1, 200:], oo[idx,t_2, 200:])
                s[t_2,t_1] = temp[0,1]

        s = np.abs(s) # (6,100) absolute correlation coefficients

        s_ave_1 = np.mean(np.abs(s[0:3, :]), axis=0)
        s_ave_2 = np.mean(np.abs(s[3:6, :]), axis=0)
        
        temp_sl = np.array([s_ave_1, s_ave_2]) # (2,100)
        sl.append(temp_sl)
    sl = np.array(sl) # (batch_size, 2, 100)
    sl = np.mean(sl, axis=0) # (2, 100) average over the batch

    return div_ev(sl)

def calR(model, node1, node2):
    '''Calculate the modularity of the graph based on the model's recurrent layer output.
    Args:
        model: The trained model.
        node1: First node for modularity calculation.
        node2: Second node for modularity calculation.
        Returns:
        Modularity value of the graph.
     '''
    
    G_r = nx.Graph()

    l = model.layers[0]
    omomi_r = np.array(l.get_weights()[1]) # get recurrent weights

    omomi_r = np.abs(omomi_r[:,200:300]) # take only the second half of the weights

    omomi_r_t = omomi_r.transpose()

    o_r = (omomi_r + omomi_r_t) / 2

    for i in range(1,100 + 1):
        G_r.add_nodes_from([str(i)])

    for j in range(100):
        for k in range(j+1,100):
            G_r.add_weighted_edges_from([(str(j+1), str(k+1), o_r[j,k])])

    # print(nx.community.modularity(G, communities=[node1,node2]))
    return nx.community.modularity(G_r, communities=[node1,node2]) # type:ignore

def calcQ_from_weight(W, node1=None, node2=None):
    '''Calculate the modularity of the graph based on the model's recurrent layer output.
    Args:
        model: The trained model.
        node1: list of nodes index for modularity calculation.
        node2: list of nodes index for modularity calculation.
        Returns:
        Modularity value of the graph.
     '''
    
    N = W.shape[0]

    if node1 is None or node2 is None:
        node1 = [str(i) for i in range(N//2)]
        node2 = [str(i) for i in range(N//2, N)]

    group_list = [node1, node2]

    aW = np.abs(W)
    aW_sym = aW + aW.transpose() # make the matrix symmetric    
    # remove self-connections
    np.fill_diagonal(aW_sym, 0)
    G = nx.from_numpy_array(aW_sym) # create a graph from the symmetric matrix

    return  nx.algorithms.community.modularity(G,group_list)

###

def calc_Q_and_D(model, x_test_m, node1, node2, batch_size=50):
    '''Calculate modularity and separation indices for the model.'''
    Q_cor = calM(model, x_test_m, node1, node2, batch_size=batch_size)
    Q_str = calR(model, node1, node2)
    D_cor = calS(model, x_test_m, batch_size=batch_size)
    D_out = calD(model)
    return Q_cor, Q_str, D_cor, D_out

###
def community_analysis(W):
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities
    a_W = np.abs(W)  # absolute correlation matrix
    N = a_W.shape[0]
    G = nx.Graph()
    N = W.shape[0]
    pos = nx.spring_layout(G)

    for i in range(1,N + 1):
        G.add_nodes_from([str(i)])

    for j in range(N):
        for k in range(j+1,N):
            G.add_weighted_edges_from([(str(j+1), str(k+1), a_W[j,k])])

    # nx.draw(G, with_labels = True)    

    # set colors for the communities

    colors = ['skyblue', 'maroon', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'olive']
    pos = nx.spring_layout(G)

    # community detection algorithm. Number of communities is set to 2.

    lst_b = greedy_modularity_communities(G, weight='weight', cutoff=2, best_n=2)
    color_map_b = ['black'] * nx.number_of_nodes(G)
    counter = 0
    for c in lst_b: #type: ignore
        for n in c:
            color_map_b[int(n)-1] = colors[counter]
        counter = counter + 1

    # パラメータ設定
    edge_percentile = 50  # draw only the top X% edges by weight（100-edge_percentile=X）
    min_edge_width = 0.2  
    max_edge_width = 2.0  
    node_size = 400      
    edge_alpha = 0.4  

    plt.figure(figsize=(15,10))

    # get edge weights and filter edges based on the specified percentile
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    threshold = np.percentile(edge_weights, edge_percentile)
    filtered_edges = [(u, v) for u, v in G.edges() if G[u][v]['weight'] >= threshold]
    filtered_weights = [G[u][v]['weight'] for u, v in filtered_edges]


    # edge width
    if filtered_weights:
        min_weight = min(filtered_weights)
        max_weight = max(filtered_weights)
        edge_widths = [min_edge_width + (max_edge_width - min_edge_width) * 
                        (w - min_weight) / (max_weight - min_weight) for w in filtered_weights]
    else:
        edge_widths = []


    # set edges
    nx.draw_networkx_edges(G, pos, edgelist=filtered_edges, width=edge_widths, alpha=edge_alpha) #type:ignore
    # set nodes
    nodes_1_50 = [str(i) for i in range(1, N//2+1)]
    nodes_51_100 = [str(i) for i in range(N//2+1, N+1)]

    # colors
    colors_1_50 = [color_map_b[i-1] for i in range(1, N//2+1)]
    colors_51_100 = [color_map_b[i-1] for i in range(N//2+1, N+1)]


    nx.draw_networkx_nodes(G, pos, nodelist=nodes_1_50, 
                            node_color=colors_1_50, node_shape='o', node_size=node_size) #type:ignore
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_51_100, 
                            node_color=colors_51_100, node_shape='^', node_size=node_size) #type:ignore


    # Labels
    nx.draw_networkx_labels(G, pos=pos, font_size=8)

    # plt.title('Network Visualization\n(Circles: nodes 1-50, Triangles: nodes 51-100)\n(Top 30% edges by weight)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


    from sklearn.metrics import  normalized_mutual_info_score
    # similarity (NMI) between the true community structure and predicted community structure
    com_true = np.zeros(N, dtype=int)
    com_true[:N//2] = 1  # 1-50 for community 1
    com_true[N//2:] = 2  # 51-100 for community 2

    com_pred = np.zeros(N, dtype=int)
    for i, c in enumerate(lst_b): #type:ignore
        for n in c: # type:ignore
            com_pred[int(n)-1] = i + 1  
    # current figure object 
    fig = plt.gcf()
    # print(com_true)
    # print(com_pred)
    nmi = normalized_mutual_info_score(com_true, com_pred)
    # print(f"normalized mutual information: {nmi}")
    return nmi, fig

###

def r2_score(y_true, y_pred):
    """Calculate the R-squared scores for each variables.
    Args:
        y_true: True values. shape (batch_size, time_steps, num_outputs)
        y_pred: Predicted values."""
    ss_res = np.sum((y_true - y_pred) ** 2, axis=(0,1)) # (num_outputs,)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=(0,1))) ** 2, axis=(0,1)) # (num_outputs,)
    return 1 - (ss_res / ss_tot)

######

class WeightRecorder:
    '''Record and restore GRU model weights during training'''

    def __init__(self):

        self.steps = []
        self.rec_kernels = []  # recurrent_kernel weights
        self.kernels = []      # input kernel weights  
        self.biases = []       # bias weights
        self.output_weights = []  # output layer weights
        self.output_biases = []   # output layer biases

    def record(self, model, step):
        '''Record only the GRU cell weights (recurrent_kernel, kernel, and bias), not the entire model's weights.
        Args: 
            model: GRU model. The model should have a GRU cell with recurrent_kernel, kernel, and bias attributes.
            step: Current step number.
        '''
        self.steps.append(step)
        self.rec_kernels.append(model.gru.cell.recurrent_kernel.numpy())
        self.kernels.append(model.gru.cell.kernel.numpy())
        self.biases.append(model.gru.cell.bias.numpy())
        self.output_weights.append(model.layers[-1].get_weights()[0])
        self.output_biases.append(model.layers[-1].get_weights()[1])

    def restore_weights(self, model, ind=-1):
        '''
        Restore model weights from recorded state, overwriting the model's weights in-place.

        Args: 
            model: GRU model to restore weights to. The model's weights will be overwritten in-place.
            ind: Index of weight record to restore (default: latest)
        '''
        model.gru.cell.recurrent_kernel.assign(self.rec_kernels[ind])
        model.gru.cell.kernel.assign(self.kernels[ind])
        model.gru.cell.bias.assign(self.biases[ind])
        model.layers[-1].set_weights([self.output_weights[ind], self.output_biases[ind]])

