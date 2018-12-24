import networkx as nx


def get_G_from_edges(edges):
    edge_dict = dict()
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    # tmp_G = nx.DiGraph()
    tmp_G = nx.Graph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        x = edge_key.split('_')[0]
        y = edge_key.split('_')[1]
        tmp_G.add_edge(x, y)
        tmp_G[x][y]['weight'] = weight
    return tmp_G


def load_network_data(f_name):
    # This function is used to load multiplex data
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_edges.append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    # create common layer.
    all_edges = list(set(all_edges))
    edge_data_by_type['Base'] = all_edges
    print('total nodes: ' + str(len(all_nodes)))
    print('Finish loading data')
    return edge_data_by_type, all_edges, all_nodes