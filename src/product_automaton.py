import time
import json
from queue import Empty
from pyTWTL import lomap
import pyTWTL.synthesis as synth
import networkx as nx
import math
import numpy as np
import random
from tqdm import tqdm
import os
from scipy.optimize import linprog

class AugPa(lomap.Model):

    def __init__(self, aug_mdp, dfa, time_bound, width, height, dfa_kind, critical_time, des_prob):
        # aug_mdp is an augmented mdp such as a tau-mdp or flag-mdp
        # dfa is generated from a twtl constraint
        # time_bound is both the time bound of the twtl task, and the time horizon of the STL constraint

        lomap.Model.__init__(self, directed=True, multi=False)

        self.aug_mdp = aug_mdp
        self.dfa = dfa
        self.kind = dfa_kind
        self.time_bound = time_bound
        self.reward_cache = {}
        self.is_STL_objective = not (aug_mdp.name == 'Static Reward MDP')
        self.width = width
        self.height = height
        self.std_ql = False
        self.critical_time = critical_time
        self.sub_des_prob = {i:des_prob**(1/len(self.critical_time)) for i in self.critical_time}
        self.sub_init_value = {i:0 for i in self.critical_time}
        product_model,_,_ = synth.modified_ts_times_fsa(aug_mdp, dfa, self.time_bound)
        self.init_dict = product_model.init
        self.g = product_model.g
        self.plot_graph(self.dfa.g,'DFA')
        self.final = product_model.final
        self.idx_to_action = {0:'stay',1:'E',-1:'W',-self.width:'N',self.width:'S',
                              -self.width-1:'NW',-self.width+1:'NE',self.width-1:'SW',self.width+1:'SE'}
        self.action_to_idx = {'stay':0,'E':1,'W':-1,'N':-self.width,'S':self.width,
                              'NW':-self.width-1,'NE':-self.width+1,'SW':self.width-1,'SE':self.width+1}
        if self.width < 3 and self.height < 3:
            self.correct_action_flag = True
            self.corrected_action_left = {'SW':'E','W':'NE'}
            self.corrected_action_right = {'E':'SW','NE':'W',}
        else:
            self.correct_action_flag = False

        # TODO: reset_init seems like a messy thing to do
        aug_mdp.reset_init()
        aug_mdp_init = list(aug_mdp.init.keys())[0]
        dfa_init = list(dfa.init.keys())[0]

        # May need to remove a certain aug mdp state
        to_remove = aug_mdp.get_state_to_remove()
        pa_to_remove = [p for p in self.get_states() if self.get_aug_mdp_state(p) == to_remove]
        self.g.remove_nodes_from(pa_to_remove)

        self.init = {p_s:1 for p_s in self.init_dict.keys() if p_s[0]==aug_mdp_init}

        # Generate set of null states
        self.null_states = self._gen_null_states()

        # allow caller to compute energy so it can be timed
        self.energy_dict = None

    def plot_graph(self, graph, fname):
        uwnew = nx.nx_agraph.to_agraph(graph)
        uwnew.layout(prog='dot')
        path = 'output/DFA-outputs'
        if not os.path.isdir(path):
            os.mkdir(path)
        uwnew.draw('output/DFA-outputs/{}.png'.format(fname))
        print('done')

    def _gen_null_states(self):
        null_pa_states = []
        for aug_mdp_s in self.aug_mdp.g.nodes():
            null_aug_mdp_s = self.aug_mdp.get_null_state(self.aug_mdp.get_null_state(aug_mdp_s))
            ts_prop = self.aug_mdp.g.nodes[null_aug_mdp_s].get('prop',set())
            fsa_state = self.dfa.next_states_of_fsa(list(self.dfa.init.keys())[0], ts_prop)[0]
            null_pa_s = (null_aug_mdp_s, fsa_state)
            null_pa_states.append(null_pa_s)
        return null_pa_states

    def get_null_states(self):
        return self.null_states

    def get_aug_mdp_state(self, pa_state):
        return pa_state[0]

    def get_mdp_state(self, pa_state):
        aug_mdp_state = pa_state[0]
        return self.aug_mdp.get_mdp_state(aug_mdp_state)

    def get_dfa_state(self, pa_state):
        dfa_state = pa_state[1]
        return dfa_state

    def get_hrz(self):
        return self.aug_mdp.get_hrz()

    def get_states(self):
        return self.g.nodes()

    def get_tpa_state_size(self):
        return len(self.g.nodes()) * self.time_bound

    def compute_energy(self):

        # Decrease compute time significantly by computing energy over pa of simple mdp and dfa
        #   and then projecting to this pa

        mdp = self.aug_mdp.get_mdp()
        # Following is O(nda*2^|AP|)
        simple_pa = synth.ts_times_fsa(mdp, self.dfa)

        # make a virtual node as the end point using 0 for weight
        simple_pa.g.add_edges_from([(p, 'virtual', {'weight':0}) for p in simple_pa.final])

        # compute minimum path costs (energy)
        # NOTE: the simple_pa does not have weights. ts_times_fsa does not carry them over. ts_times_fsa must be reimplemented if weights on the PA are desired.
        #       I specify the 'weight' attribute anyways for future sake.
        # It seems that any node that cannot reach target is excluded from the returned dict. It does not raise an error as the docs state.
        # complexity of dijkstra using min-priority que is O((v+e)log(v)) with verticies v and edges e. This can be simplified to O(e*log(v)) if the graph is connected.
        # O(nad*log(nd)). With nd nodes, nad edges, because the product MDP uses the MDP transition function with edges n*a, applied d times.
        simple_energy_dict = nx.shortest_path_length(simple_pa.g, target='virtual', weight='weight')

        # project onto full PA
        energy_dict = {}
        for p in self.get_states():
            mdp_s = self.get_mdp_state(p)
            dfa_s = self.get_dfa_state(p)
            try:
                energy_dict[p] = simple_energy_dict[(mdp_s, dfa_s)]
            except KeyError:
                energy_dict[p] = float('inf')
        self.energy_dict = energy_dict
        nx.set_node_attributes(self.g, energy_dict, name='energy')

    def get_energy(self, pa_state):
        return self.energy_dict[pa_state]

    def time_product_mdp(self):     
        ep_len = self.time_bound
        ygraph,pp,new_TPMDP = synth.modified_ts_times_fsa(self.aug_mdp, self.dfa, self.time_bound)
        copy_new_graph = new_TPMDP.copy()
        tpmdp_transitions = nx.convert.to_dict_of_lists(copy_new_graph)
        self.newg = copy_new_graph

        return tpmdp_transitions
    
    def prune_actions(self, eps_uncertainty, des_prob):  
        self.back_propagation(eps_uncertainty)
        max_value_list = list(self.sub_init_value.values())
        sub_prob_list = list(self.sub_des_prob.values())
        if sum(list(np.array(max_value_list) - np.array(sub_prob_list) >= 0)) < len(self.critical_time):
            print("assumption not hold")
            time.sleep(2)
                    
        ep_len = self.time_bound
        self.pi_c = {t:{} for t in range(ep_len + 1)}
        # initialize actions of time product MDP
        pruned_states_tpmdp = [nx.convert.to_dict_of_lists(self.newg) for _ in range(ep_len + 1)]       
        pruned_actions_tpmdp = {t:{} for t in range(ep_len + 1)}
        for t_p in self.newg.nodes():
            pruned_actions_tpmdp[t_p[2]][t_p[0:2]]=[]
        
        pruned_actions = pruned_actions_tpmdp
        pruned_states = []
        for i in pruned_states_tpmdp:          
            dic = dict()
            for s in i:
                if s[1] != "trash":
                    dic[s[:2]] = [j[:2] for j in i[s]]
                    
                else:
                    dic[s[:2]] = ["trash"]
            pruned_states.append(dic)

        # create set of non-accepting states
        accepting_states = []
        for n in self.newg.nodes:
            if n[1] in self.dfa.final:
                accepting_states.append(n)
        
        non_accepting_states = list(pruned_states_tpmdp[0].keys())
        for s in pruned_states_tpmdp[0].keys():
            if s[1] in self.dfa.final:
                non_accepting_states.remove(s)

        for ss in accepting_states:
            t = ss[2]
            p = ss[:2]
            test_neighbors = self.aug_mdp.g.neighbors(p[0])
            test_neighbors = [(i,0) for i in test_neighbors]
            pruned_actions[t][p] = [self.states_to_action(p,q) for q in test_neighbors]
            self.pi_c[t][p] = self.final_sa_values[ss][1]

        for pp in tqdm(non_accepting_states):
            t = pp[2]
            p = pp[:2]
            if self.std_ql:
                test_neighbors = self.aug_mdp.g.neighbors(p[0])
                test_neighbors = [(i,0) for i in test_neighbors]
                pruned_actions[t][p] = [self.states_to_action(p,q) for q in test_neighbors]

            else:
                self.pi_c[t][p] = self.final_sa_values[pp][1]
                if pp[1] != "trash" and self.opt_s_value[pp]!=0:
                    next_ps = pruned_states[t][p][:]
                    for next_p in next_ps:
                        next_states = self.get_low_prob_neighbors(p,next_p)
                        next_states.append(next_p)
                        for nn in next_states:
                            if t+1 not in self.critical_time:
                                next_value = self.final_sa_values[nn+(t+1,)][0]
                            else:
                                next_value = int(self.opt_s_value[nn+(t+1,)]>=self.sub_des_prob[t+1])
                            if next_value < des_prob**(1/len(self.critical_time)):
                                pruned_states[t][p].remove(next_p)
                                break
                    pruned_actions[t][p] = [self.states_to_action(p,q) for q in pruned_states[t][p]]
                else:
                    test_neighbors = self.aug_mdp.g.neighbors(p[0])
                    test_neighbors = [(i,0) for i in test_neighbors]
                    pruned_actions[t][p] = [self.states_to_action(p, neighbor) for neighbor in test_neighbors]

        for pp in self.newg.nodes():
            p = pp[:2]
            test_neighbors = self.aug_mdp.g.neighbors(p[0])
            test_neighbors = [(i,0) for i in test_neighbors]
            if p+(ep_len,) in self.newg.nodes():
                pruned_actions[ep_len][p] = [self.states_to_action(p, neighbor) for neighbor in test_neighbors]
        
        self.pruned_states = pruned_states
        self.pruned_actions = pruned_actions
        test_dict = {}
        for t in self.pruned_actions:
            test_dict[t] = {str(i):self.pruned_actions[t][i] for i in self.pruned_actions[t]}
            
        with open('data/prune_actions.txt', 'w') as convert_file:
            convert_file.write(json.dumps(test_dict))

    def back_propagation(self, eps_uncertainty):
        tpa_transitions = self.time_product_mdp()

        accepting_states = self.final
        
        All_Actions = dict()
        
        for t in range(self.time_bound):
            for tpa_s in tpa_transitions:
                if tpa_s[2] == t:
                    s = tpa_s[:2]
                    All_Actions[tpa_s] = dict()
                    available_actions = self.all_possible_actions(s)
                    for i in range(len(available_actions)):
                        a = available_actions[i]
                        OME_state = self.all_possible_transition(s, a)
                        # Choose next state from possible low probability states
                        epsilon_transitions = [p+(t+1,) for p in self.get_low_prob_neighbors(s,OME_state)]
                        epsilon_transitions.append(OME_state+(t+1,))
                        All_Actions[tpa_s][i] = (available_actions[i],epsilon_transitions)         

        final_opt_value_sa = dict()

        Layer_accepting_states=[]
        self.tpa_layer_accepting_states = Layer_accepting_states
        
        opt_sa_value = dict()
        self.opt_s_value = dict()

        for tt in tqdm(range(self.time_bound+1)):
            t = self.time_bound - tt
            for tpa_s in tpa_transitions:
                multi_action_maximizer = []
                if tpa_s[2] == t:
                    opt_sa_value[tpa_s]=dict()
                    if (tpa_s not in Layer_accepting_states) and (tpa_s[1] not in self.dfa.final) and (tpa_s[1] != "trash"):
                        self.opt_s_value[tpa_s] = 0
                        for i in All_Actions[tpa_s]:                         
                            chosen_action = All_Actions[tpa_s][i][0]
                            self.tpa_opt_temprory_states = All_Actions[tpa_s][i][1]
                            opt_sa_value[tpa_s][chosen_action] = self.sa_pair_opt(eps_uncertainty, tpa_s, chosen_action)
                            
                            if opt_sa_value[tpa_s][chosen_action] >= self.opt_s_value[tpa_s]:
                                self.opt_s_value[tpa_s] = opt_sa_value[tpa_s][chosen_action]
                        
                        for i in All_Actions[tpa_s]:                         
                            chosen_action = All_Actions[tpa_s][i][0]
                            if opt_sa_value[tpa_s][chosen_action] == self.opt_s_value[tpa_s]:
                                multi_action_maximizer.append(chosen_action)                           
                        final_opt_value_sa[tpa_s] = (self.opt_s_value[tpa_s],multi_action_maximizer)
                        
                    elif ((tpa_s in Layer_accepting_states) or (tpa_s[1] in self.dfa.final)):
                        self.opt_s_value[tpa_s] = 1

                    elif (tpa_s[1] == "trash"):
                        self.opt_s_value[tpa_s] = 0
        for tpa_s in tpa_transitions:
            if (tpa_s[1] == "trash"):
                final_opt_value_sa[tpa_s] = (0.0,"true")
            elif (tpa_s[1] in self.dfa.final):
                final_opt_value_sa[tpa_s] = (1.0,"true")

        self.final_sa_values = final_opt_value_sa
        self.opt_sa_value = opt_sa_value 

    def sa_pair_opt_value(self, s):
        if s[2] not in self.critical_time:
            return self.opt_s_value[s]
        else:
            t_value = int(self.opt_s_value[s]>=self.sub_des_prob[s[2]])    
            if self.opt_s_value[s] > self.sub_init_value[s[2]]:
                self.sub_init_value[s[2]] = self.opt_s_value[s]              
            return t_value

    def sa_pair_opt(self, eps_uncertainty, s, a):
        obj = []
        for i in self.tpa_opt_temprory_states:
            obj.append(self.sa_pair_opt_value(i))

        if sum(obj) == 0:
            return 0
        else:
            lhs_eq = [[1 for i in range(len(obj))]]
            rhs_eq = [1]       
            lhs_ineq = [[0 for i in range(len(obj)-1)]+[1],
                        [0 for i in range(len(obj)-1)]+[-1],
                        [1 for i in range(len(obj)-1)]+[0],
                        [-1 for i in range(len(obj)-1)]+[0],]
            rhs_ineq = [1, eps_uncertainty-1, 1-eps_uncertainty, 0]
            bnd = []
            for b in range(len(obj)):
                bnd.append((0, float("inf")))
            opt = linprog(c=obj, A_ub=lhs_ineq, b_ub=rhs_ineq, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method="revised simplex")
            return opt.fun

    def states_to_action(self, s1, s2): 
        match self.aug_mdp.name:
            case "Static Reward MDP":
                idx1 = int(s1[0][1:])
                idx2 = int(s2[0][1:])
            case "Tau MDP":
                idx1 = int(s2[0][self.tau-2][1:])
                idx2 = int(s2[0][self.tau-1][1:])
            case "Flag MDP":
                idx1 = int(s1[0][0][1:])
                idx2 = int(s2[0][0][1:])
        action = self.idx_to_action[idx2-idx1]
        if self.correct_action_flag:
            if action in self.corrected_action_left and (idx1==0 or idx1==2):
                action = self.corrected_action_left[action]
            elif action in self.corrected_action_right and (idx1==1 or idx1==3):
                action = self.corrected_action_right[action]           
        return action        

    def take_action(self, s, a, uncertainty):
        # Action is being defined as a state-state transition. 
        #   Possibly use discrete actions in the future in the case that multiple states can
        #   result from a single action with significant probability.
        
        # for verifying next_state
        # next_state = [i for i in self.pruned_states[t][s] if int(i[0][1:])==next_idx][0]
        match self.aug_mdp.name:
            case "Static Reward MDP":
                cur_idx = int(s[0][1:])
                next_idx = cur_idx + self.action_to_idx[a]
                next_aug_mdp_s = 'r{}'.format(next_idx)
            case "Tau MDP":
                cur_idx = int(s[0][self.tau-1][1:])
                next_idx = cur_idx + self.action_to_idx[a]
                next_mdp_s = 'r{}'.format(next_idx)
                next_aug_mdp_s = s[0][1:]+(next_mdp_s,)
            case "Flag MDP":
                cur_idx = int(s[0][0][1:])
                next_idx = cur_idx + self.action_to_idx[a]
                next_mdp_s = 'r{}'.format(next_idx)
                flags = s[0][1]
                next_flags = self.aug_mdp.fmdp_stl.flag_update(flags, next_mdp_s)
                next_aug_mdp_s = (next_mdp_s,next_flags)
        ts_prop = self.aug_mdp.g.nodes[next_aug_mdp_s].get('prop',set())
        fsa_next_state = self.dfa.next_states_of_fsa(s[1], ts_prop)[0]
        next_s = (next_aug_mdp_s, fsa_next_state)

        if np.random.uniform() > uncertainty:
            return next_s
        else:
            # Choose next state from possible low probability states
            low_prob_states = self.get_low_prob_neighbors(s,next_s)
            if low_prob_states == []:
                # if no low prob states exist, action must be stay and s2 is the only option
                return next_s

            # Choose from low probability states
            next_s = random.choice(low_prob_states)
            
            return next_s

    def get_low_prob_neighbors(self, s1, s2):
        # TODO: put some of this in an mdp class
        # region_to_xy = self.aug_mdp.sig_dict
        # TODO should be more generalized than using sig_dict
        region_to_xy = {r:(d['x'],d['y']) for r,d in self.aug_mdp.sig_dict.items()}
        neighbors = self.g.neighbors(s1)
        xy_to_pa = {region_to_xy[self.get_mdp_state(pa_s)]:pa_s for pa_s in neighbors}

        mdp1 = self.get_mdp_state(s1)

        x,y = region_to_xy[mdp1]

        # Map directions to next states
        # This may include "out of bounds" states
        adj_xy_dict = {
            'N': (x-1,y),
            'NE':(x-1,y+1),
            'E': (x,y+1),
            'SE':(x+1,y+1),
            'S': (x+1,y),
            'SW':(x+1,y-1),
            'W': (x,y-1),
            'NW':(x-1,y-1),
            'stay':(x,y)
        }

        # convert xy to pa state and drop invalid
        adj_dict = {}
        for k in adj_xy_dict:
            try:
                adj_dict[k] = xy_to_pa[adj_xy_dict[k]]
            except KeyError:
                pass

        adj_dict_inv = {adj_dict[k]:k for k in adj_dict}
        dir_list = ['N','NE','E','SE','S','SW','W','NW']

        chosen_dir = adj_dict_inv[s2]
        if chosen_dir == 'stay':
            # No low probability states for this action
            return []

        low_prob_states = []
        chosen_idx = dir_list.index(chosen_dir)
        for i in [-1, 1]:
            alt_dir = dir_list[(chosen_idx + i) % len(dir_list)]
            try:
                low_prob_states.append(adj_dict[alt_dir])
            except:
                # alt_dir is out of the environment
                pass
        low_prob_states.append(adj_dict['stay'])  # stay

        return low_prob_states

    def initial_state_and_time(self, init_pa_state = None):
        # assume agent has remained in initial mdp state for timesteps 0 thru tau - 2
        # also give initial time that rewards are summed over in the q-learning problem formulation
        if init_pa_state == None:
            z = list(self.init.keys())[0]
        else:
            z = init_pa_state
            if z not in self.get_states():
                raise Exception("invalid pa state: {}".format(z))
        tau = self.aug_mdp.get_tau()
        init_traj = [z]

        for _ in range(1,tau):
            mdp_s = self.get_mdp_state(z)
            neighbors = self.g.neighbors(z)
            # choose next z with same mdp state
            z = next(iter([next_z for next_z in neighbors if self.get_mdp_state(next_z) == mdp_s]))
            init_traj.append(z)
        
        t_init = tau-1
        return z, t_init, init_traj

    def reward(self, pa_s, beta = 2):        
        if not self.std_ql:
            aug_mdp_s = pa_s[0]
            try:
                rew = self.reward_cache[(aug_mdp_s,beta)]
            except KeyError:
                if not self.aug_mdp.is_state(aug_mdp_s):
                    raise Exception("Invalid augmented mdp state!")
                rew = self.aug_mdp.reward(aug_mdp_s, beta)
                self.reward_cache[(aug_mdp_s,beta)] = rew
            return rew
        else:
            if pa_s[1] in self.dfa.final:
                return 1
            else:
                return 0

    def get_null_state(self, pa_s):

        aug_mdp_s = self.get_aug_mdp_state(pa_s)
        null_aug_mdp_s = self.aug_mdp.get_null_state(aug_mdp_s)
        ts_prop = self.aug_mdp.g.nodes[null_aug_mdp_s].get('prop',set()) 
        fsa_state = self.dfa.next_states_of_fsa(list(self.dfa.init.keys())[0], ts_prop)[0]
        null_pa_s = (null_aug_mdp_s, fsa_state)

        if null_pa_s not in self.get_states():
                raise Exception('Error: invalid null state: {}'.format(null_pa_s))

        return null_pa_s

    def is_accepting_state(self, pa_s):

        dfa_state = self.get_dfa_state(pa_s)

        return dfa_state in self.dfa.final

    def sat(self, pa_s):

        aug_mdp_s = self.get_aug_mdp_state(pa_s)

        return self.aug_mdp.sat(aug_mdp_s)

    def gen_new_ep_states(self):
        # Time complexity is O(x*n^r)
        # Generates a dictionary that maps each pa state (at end of ep) to possible pa states at t = tau-1, and each of those 
        #   PA states to possible initial trajectories that could lead to that
        # Each new episode starts with a state adjacent to the last state of the previous episode

        # Nothing to do for non STL objective
        if not self.is_STL_objective:
            return

        new_ep_dict = {}

        def new_ep_states_recurse(pa_s, tau, t = 0, temp_dict = None, hist = None):

            # Assumes that pa_s is the state chosen at t = 0 TODO: is this still correct?
            if temp_dict == None:
                temp_dict = {}
            if hist == None:
                hist = []
            else:
                hist.append(pa_s)

            try:
                neighbors = self.pruned_states[t][pa_s]
            except KeyError:
                pa_s = (pa_s[0], list(self.dfa.init.keys())[0] + 1)
                neighbors = self.pruned_states[t][pa_s]

            # for eg static rewards
            if tau == 1:
                return {n: [[n]] for n in neighbors}

            if t == tau-1:
                for n in neighbors:
                    if n in temp_dict:
                        pass # TODO: Find a less memory intensive method. Take first history until then.
                    else:
                        temp_dict[n] = [hist + [n]]
                hist.pop()
                return temp_dict
            
            for n in neighbors:
                temp_dict = new_ep_states_recurse(n, tau, t + 1, temp_dict, hist)
            if t != 0:
                hist.pop()
            return temp_dict

        tau = self.aug_mdp.get_tau()              

        for pa_s in tqdm(self.get_states()):
            if pa_s not in new_ep_dict:
                aug_mdp_s = self.get_aug_mdp_state(pa_s)
                null_aug_mdp_s = self.aug_mdp.get_null_state(aug_mdp_s)
                null_pa_s = (null_aug_mdp_s, list(self.dfa.init.keys())[0])
                if null_pa_s not in new_ep_dict:
                    new_ep_dict[null_pa_s] = new_ep_states_recurse(null_pa_s, tau)
                    if new_ep_dict[null_pa_s] == {}:
                        # TODO:
                        raise Exception('\nError: Augmented Product MDP state {} at t=0 can not reach a state at t={} '.format(null_pa_s, tau) 
                                + 'on the Pruned Augmented Time-Product MDP. This is most likely due to all potential actions being pruned. '
                                + 'It is assumed that all states at t=0 can reach a state at t=tau so that, in the case of an STL objective, '
                                + 'potential initial trajectories can be computed.\n'
                                + 'This can be resolved by either (1) reducing the over estimated action uncertainty, (2) reducing the desired '
                                + 'satisfaction probability (3) increasing the time horizon, or (4) modifying the constraint mission such that '
                                + 'it can be completed in fewer time steps.')
                new_ep_dict[pa_s] = new_ep_dict[null_pa_s]
                # TODO maybe only key on null_pa_s to save ram for large state spaces, if it is indeed copying here

        self.new_ep_dict = new_ep_dict

    def get_new_ep_states(self, pa_s):
        if not self.is_STL_objective:
            raise RuntimeError('This function should not be called for non STL objective.')
        new_ep_states = list(self.new_ep_dict[pa_s].keys())
        return new_ep_states

    def get_new_ep_trajectory(self, last_pa_s, init_pa_s):
        if not self.is_STL_objective:
            raise RuntimeError('This function should not be called for non STL objective.')
        new_ep_trajs = self.new_ep_dict[last_pa_s][init_pa_s]
        selection_idx = np.random.choice(len(new_ep_trajs))
        selection = new_ep_trajs[selection_idx]
        return selection

    def find_all_paths(self, graph, start, path=[]):

        global samsam
        path = path + [start]
        paths = [path]
        if len(graph[start]) == 0:  # No neighbors
            samsam.append(path)
        for node in graph[start]:
            newpaths = self.find_all_paths(graph, node, path)
            for newpath in newpaths:
                paths.append(newpath)
        return paths

    def all_possible_transition(self, s, a):
        # s is product_automaton state
        # Action is being defined as a state-state transition.
        #   Possibly use discrete actions in the future in the case that multiple states can
        #   result from a single action with significant probability.

        # for verifying next_state
        # next_state = [i for i in self.pruned_states[t][s] if int(i[0][1:])==next_idx][0]

        cur_idx = int(s[0][1:])
        next_idx = cur_idx + self.action_to_idx[a]
        ts_next_state = 'r{}'.format(next_idx)
        ts_prop = self.aug_mdp.g.nodes[ts_next_state].get('prop',set())
        fsa_next_state = self.dfa.next_states_of_fsa(s[1], ts_prop)[0]
        one_minus_epsilon_transition = (ts_next_state, fsa_next_state)

        return one_minus_epsilon_transition

    def all_possible_actions(self,s):
    
        # s is product_automaton state
        neighbors = self.g.neighbors(s)
        actions = [self.states_to_action(s,n) for n in neighbors]
        return actions

