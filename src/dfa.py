import networkx as nx
import os

from pyTWTL.twtl_to_dfa import twtl_to_dfa
from pyTWTL.lomap import Fsa


CONFIG_PATH = '../configs/default_static.yaml'


def create_dfa(twtl_cfg: dict, env_cfg: dict) -> tuple[Fsa, int, str]:

    n = env_cfg['width']
    def xy_to_region(x,y,z):
        # x is down, y is across
        # ignore 3rd dim
        return x * n + y

    region_coords = twtl_cfg['regions']
    custom_task = twtl_cfg['TWTL task']
    if custom_task == 'None':
        custom_task = None

    if custom_task != None:
        # Build the spec from regions in config

        for r,c in region_coords.items():
            region_num = xy_to_region(*c)
            custom_task = custom_task.replace(r, 'r' + str(region_num))
        phi = custom_task
    else:
        # pickup and delivery format
        pickup = region_coords['pickup']
        delivery = region_coords['delivery']

        pick_up_reg = xy_to_region(*pickup)
        delivery_reg = xy_to_region(*delivery)
        pick_up_str  = str(pick_up_reg)
        delivery_str = str(delivery_reg)

        twtl_horizon = twtl_cfg['time horizon']
        tf1 = int((twtl_horizon-1)/2) # time bound
        tf2 = int(twtl_horizon) - tf1 - 1
        phi = '[H^1 r' + pick_up_str + ']^[0, ' +  str(tf1) + '] * [H^1 r' + delivery_str + ']^[0,' + str(tf2) + ']'

    kind = twtl_cfg['DFA kind']
    out = twtl_to_dfa(phi, kind=kind, norm=True)
    dfa = out[kind]
    bounds = out['bounds']
    dfa_horizon = bounds[-1]
    # check horizon is as expected
    if custom_task == None and dfa_horizon != twtl_horizon:
        raise RuntimeError(f'Received unexpected time bound from DFA. DFA horizon: {dfa_horizon}, expected: {twtl_horizon}.')
    else:
        # dfa horizon and twtl horizon are the same. Good.
        pass        

    # add self edge to accepting state
    # All observation cases in accepting state should result in self edge
    input_set = dfa.alphabet    
    for s in dfa.final:
        dfa.g.add_edge(s,s, guard='(else)', input=input_set, label='(else)', weight=0)

    if (twtl_cfg['DFA modification type'] == 'total'):
        dfa_total = total_dfa(dfa)

    # String to print
    print_string = f'TWTL task: {phi}\n'
    if custom_task != None:
        for r,c in region_coords.items():
            rnum = xy_to_region(*c)
            print_string += f'{r} : {c} <---> Region {rnum}\n'
    else:
        print_string += 'Pick-up Location  : ' + str(pickup) + ' <---> Region ' + pick_up_str + '\n'
        print_string += 'Delivery Location : ' + str(delivery) + ' <---> Region ' + delivery_str + '\n'

    return dfa_total, dfa, dfa_horizon, print_string

def save_dfa(dfa: Fsa, path: str='../output/dfa.png') -> None:

    this_path = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_path, path)
    A = nx.nx_agraph.to_agraph(dfa.g)
    A.draw(path, prog='dot')


def save_custom_dfa(dfa: Fsa) -> None:

    A = nx.nx_agraph.to_agraph(dfa.g)
    ns = list(A.iternodes())
    ns[0].attr['pos'] = '0,100'
    ns[1].attr['pos'] = '150,100'
    ns[2].attr['pos'] = '250,100'
    ns[3].attr['pos'] = '250,0'
    ns[4].attr['pos'] = '100,0'
    ns[4].attr['shape'] = 'doublecircle'
    for e in A.iteredges():
        if tuple(e) == ('4','4'):
            e.attr['pos'] = 'e,78,7 78,-7 60,-7 50,-4 48,0 50,4 60,7 70,7'
    A.draw('dfa.png', prog='neato', args='-n2 -Gsplines=polyline')


def total_dfa(original_dfa):

    '''
    Convert a normal DFA to a total DFA

    Author: Abbasali Koochakzadeh
    Update: May 17th 2022
    '''

#   If you happend to have to work with TWTL spec uncomment section I & II

# ------------------------------------------------------
##      I - Extracting the DFA
# ------------------------------------------------------

    # out = twtl_to_dfa(twtl_spec, kind='normal', norm=True)
    # dfa = out['normal']
    # bounds = out['bounds']
    # dfa_horizon = bounds[-1]
    # dfa.add_trap_state()

# ------------------------------------------------------
##      II - Adding proper self loop on accepting states
# ------------------------------------------------------

    # edge_list = list(nx.convert.to_edgelist(dfa.g))

    # edge_table = []
    # for e in edge_list:
    #     dic = e[2]
    #     dic['node1'] = e[0]
    #     dic['node2'] = e[1]
    #     edge_table.append(dic)

    # edge_info = pd.DataFrame.from_records(edge_table)

    # input_set = set()
    # for s in dfa.final:
    #     k = 0
    #     for e in edge_list:
    #         if(e[0] == s):
    #             input_set = input_set.union(edge_info.input[k])
    #         k += 1
    #     dfa.g.add_edge(s, s, guard='(else)', input=dfa.alphabet.difference(input_set),
    #                    label='(else)', weight=0)

# ------------------------------------------------------
#      III - Adding trash state
# ------------------------------------------------------
    

# copying the orginal dfa inorder to make modification
    dfa = original_dfa.clone()

# Adding the trash state and its self loop to the dfa graph
    dfa.g.add_node('trash')
    dfa.g.add_edge('trash', 'trash', **{'guard': 'trap',
                                    'input': dfa.alphabet, 'label': 'trap', 'weight': 0})

# Making a dictinary of the transitions in the dfa graph
    edge_list = list(nx.convert.to_edgelist(dfa.g))
    edge_table = []
    for e in edge_list:
        dic = e[2]
        dic['node1'] = e[0]
        dic['node2'] = e[1]
        edge_table.append(dic)

# Following lines basically looks for missed labels on the outgoing transitions for each node
    for n in dfa.g:

# The 'node_n_input_set' is a set which will be the union of all labels used on the outgoing transitions from node 'n'
        node_n_input_set = set()

# The 'node_n_outgoing_index_start' and 'node_n_outgoing_index_end' are to pointer which point to start and end of a section
# of the transition dictionary. This section will include all the outgoing transitions from node n, in other words the
# 'node_n_outgoing_index_start' points to the first outgoing transition from node n and the 'node_n_outgoing_index_end' points
# to the last outgoing transition from node n based on the extracted transtion dictionary
        node_n_outgoing_index_start = 0
        node_n_outgoing_index_end = 0
        i = 0
        j = len(edge_list)
        for e in edge_list:
            if (e[0] == n):
                node_n_outgoing_index_start = i
                break
            i += 1
        for e in reversed(edge_list):
            j -= 1
            if (e[0] == n):
                node_n_outgoing_index_end = j
                break

# The following lines will extract all the labels used for outgoing transitions from node n which we know they are located between
# 'node_n_outgoing_index_start' and 'node_n_outgoing_index_end' in the transitions dictionary
        for k in range(node_n_outgoing_index_start, node_n_outgoing_index_end + 1):
            node_n_input_set = node_n_input_set.union(edge_table[k]["input"])

# At the end we add a trap transition to the trash state with a label of the set 'node_n_input_set' (missed labels).
# In order to compute the missed labels we compute the set difference between the whole set of possible labels(based on specification)
# and the set of used labels 'node_n_input_set'
        if (len(dfa.alphabet.difference(node_n_input_set)) != 0):
            dfa.g.add_edge(n, 'trash', **{'guard': 'trap', 'input': dfa.alphabet.difference(
                node_n_input_set), 'label': 'trap', 'weight': 0})


    return dfa