
from pyTWTL import lomap
import networkx as nx
import numpy as np
from STL import STL
from tqdm import tqdm

class Tmdp(lomap.Ts):
    def __init__(self, mdp, stl_expr, mdp_sig_dict):
        # O(n^tau) worst case
        lomap.Ts.__init__(self, directed=True, multi=False)

        self.name = 'Tau MDP'
        self.mdp = mdp
        self.stl_expr = stl_expr
        self.sig_dict = mdp_sig_dict
        self.tmdp_stl = TmdpStl(stl_expr, mdp_sig_dict)
        self.tau = self.tmdp_stl.get_tau()

        #add state for null history that can transition to any state
        for s in list(self.mdp.g.nodes()) + ['None']:
            self.mdp.g.add_edge('None', s, new_weight=1, weight=1)

        # make mdp init at that state for the sake of building
        real_mdp_init = self.mdp.init
        self.mdp.init = {'None':1}

        self.build_states()
        self.build_transitions()

        # reset mdp init
        self.mdp.init = real_mdp_init

        # When the pa is created, we want states with null history. Must start with all None.
        self.temp_init_state = tuple(['None'] * self.tau)
        self.init = {self.temp_init_state: 1}

        # self.wrap_dict = {s:self.gen_wrap_states_by_mdp(s) for s in self.mdp.g.nodes()}


    def get_hrz(self):
        return self.tmdp_stl.get_hrz()

    def get_tau(self):
        return self.tau

    def get_mdp_state(self, tmdp_s):
        if type(tmdp_s) != tuple or len(tmdp_s) != self.tau:
            raise Exception('invalid tau mdp state: {}'.format(tmdp_s))
        # return last state in sequence
        return tmdp_s[-1]

    def get_mdp(self):
        return self.mdp

    def new_ep_state(self, tmdp_s):
        new_s = ['None'] * (self.tau - 1) + [tmdp_s[-1]]
        return tuple(new_s)

    def reset_init(self):
        self.g.remove_node(self.temp_init_state)
        self.init = {(('None',) * (self.tau-1)) + (list(self.mdp.init.keys())[0],) :1}

    def get_state_to_remove(self):
        return self.temp_init_state

    # def get_wrap_states(self, tmdp_s):
    #     mdp_s = self.get_mdp_state(tmdp_s)
    #     return self.wrap_dict[mdp_s]

    def build_states(self):
        # O(n*a^(tau-1)) worst case
        # Make a dictionary of ts edges 
        ts_edge_dict = {s:list(self.mdp.g.adj[s].keys()) for s in list(self.mdp.g.adj.keys())}
        # ts_edge_dict[None] = self.mdp.g.edge.keys() + [None]

        # make list of tau mdp states where each state is represented by a tuple of mdp states
        tau = self.tmdp_stl.get_tau()
        states = []
        for s in tqdm(list(ts_edge_dict.keys())):
            states.extend(  self.build_states_recurse([s], ts_edge_dict, tau))

        # states.remove((None,) * tau) # No state should end with a null

        # try and recreate process used in ts.read_from_file() except with tau mdp
        self.init = {(('None',) * (tau-1)) + (list(self.mdp.init.keys())[0],) :1}
        self.states = states
        self.ts_edge_dict = ts_edge_dict

    def build_states_recurse(self, past, ts_edge_dict, tau):
        if tau == 1:
            # One tau-MDP state per MDP state
            return [tuple(past)]
        next_states = ts_edge_dict[past[-1]]
        if len(next_states) == 0:
            # no next states. Maybe an obstacle?
            return []
        tmdp_states = [past + [ns] for ns in next_states]
        if len(tmdp_states[0]) == tau:
            # each tau-MDP state has 'tau' MDP states
            # make each state immutable
            return [tuple(s) for s in tmdp_states]
        
        # recurse for each state in states
        more_tmdp_states = []
        for x in tmdp_states:
            more_tmdp_states.extend(self.build_states_recurse(x, ts_edge_dict, tau))
        
        return more_tmdp_states


    def build_transitions(self):
        # O(n*a) worst case

        # create dict of dicts representing edges and attributes of each edge to construct the nx graph from
        # attributes are based on the mdp edge between the last (current) states in the tau mdp sequence
        tau = self.tmdp_stl.get_tau()
        edge_dict = {}
        for x1 in tqdm(self.states):
            edge_attrs = self.mdp.g.adj[x1[-1]]
            # tmdp states are adjacent if they share the same (offset) history. "current" state transition is implied valid 
            # based on the set of names created
            if tau > 1:
                # edge_dict[x1] = {x2:edge_attrs[x2[-1]] for x2 in self.states if x1[1:] == x2[:-1]}
                edge_dict[x1] = {x1[1:]+(s2,):edge_attrs[s2] for s2 in self.mdp.g.neighbors(x1[-1])}
            else:
                # Case of tau = 1
                edge_dict[x1] = {(x2,):edge_attrs[x2] for x2 in self.ts_edge_dict[x1[0]]}

        self.g = nx.from_dict_of_dicts(edge_dict, create_using=nx.MultiDiGraph()) 

        # add node attributes based on last state in sequence
        for n in self.g.nodes():
            # self.g.nodes[n] = self.mdp.g.nodes[n[-1]]
            nx.set_node_attributes(self.g, {n: self.mdp.g.nodes[n[-1]]})

    # def gen_wrap_states_by_mdp(self,mdp_s):
    #     #TODO what about pruned time product??
    #     tmdp_s = tuple([None] * (self.tau-1) + [mdp_s])
    #     return self.gen_wrap_states(tmdp_s)

    # def gen_wrap_states(self, tmdp_s, depth = 2):
    #     # This assumes that the last mdp state of the previous episode is the first state of this episode.
    #     neighbors = self.g.neighbors(tmdp_s)
    #     if depth == self.tau:
    #         return neighbors
        
    #     final_tmdp_s_list = []
    #     for s in neighbors:
    #         tmdp_s_list = self.gen_wrap_states(s, depth + 1)
    #         final_tmdp_s_list.extend(tmdp_s_list)

    #     return final_tmdp_s_list


    def reward(self, tmdp_s, beta):
        temporal_op = self.tmdp_stl.get_outer_temporal_op()
        if temporal_op == 'F':
            r = np.exp(beta * self.tmdp_stl.rdegree_rew(tmdp_s))
        elif temporal_op == 'G':
            r = -1 * np.exp(-1 * beta * self.tmdp_stl.rdegree_rew(tmdp_s))
            positive_offset = np.exp(0)
            r += positive_offset
        return r

    def sat(self, tmdp_s):
        is_sat = (self.tmdp_stl.rdegree_rew(tmdp_s) > 0)
        return is_sat

    def is_state(self, tmdp_s):
        # TODO: potentially make this a more robust check
        return type(tmdp_s) == tuple

    def get_null_state(self, tmdp_s):
        mdp_s = self.get_mdp_state(tmdp_s)
        null_tmdp_s = tuple(['None'] * (self.tau-1) + [mdp_s])
        return null_tmdp_s


class TmdpStl:
    def __init__(self, stl_expr, ts_sig_dict):
        # O(len(stl_expr))? about O(1)
        self.expr = stl_expr
        self.ts_sig_dict = ts_sig_dict
        # self.rdegree_lse_cache = {}  # cache dict

        # extract the big phi
        end = stl_expr.index(']') + 1
        self.big_phi = stl_expr[:end]

        # extract the small phi
        self.small_phi = stl_expr[end:]

        bracket = self.big_phi.index(']')
        comma = self.big_phi.index(',')
        outer_b = int(self.big_phi[comma+1:bracket])

        bracket = self.small_phi.index(']')
        comma = self.small_phi.index(',')
        inner_hrz = int(self.small_phi[comma+1:bracket])
        self.tau = inner_hrz + 1
        self.hrz = outer_b + inner_hrz

        self.parser = STL(self.small_phi)

    def get_outer_temporal_op(self):
        return self.big_phi[0]

    # def set_ts_sig_dict(self, ts_sig_dict):
    #     self.ts_sig_dict = ts_sig_dict

    def rdegree_rew(self,tmdp_s):
        # # Create signal as coordinate position of each state in the history
        # if self.ts_sig_dict == None:
        #     raise Exception("State to signal mapping must be set with set_ts_sig_dict().")

        if 'None' in tmdp_s:
            raise RuntimeError('Error: Trying to calculate the reward for a state with incomplete history!!')
        sig = [self.ts_sig_dict[x] for x in tmdp_s]
        rdeg = self.parser.rdegree(sig)
        return rdeg


    # def rdegree_lse(self, tmdp_s):

    #     if None in tmdp_s:
    #         raise Exception("State does not have complete history. Cannot compute robustness degree of incomplete state.")

    #     # check cache
    #     try:
    #         return self.rdegree_lse_cache[tmdp_s]
    #     except KeyError:
    #         # not in cache so keep going
    #         pass

    #     # Using log-sum-exp approximation, sign is positive for F (max) and negative for G (min)
    #     if self.expr[0] == 'F':
    #         sign = 1
    #     elif self.expr[0] == 'G':
    #         sign = -1
    #     else:
    #         # expression should start with either globally (G) or eventually (F)
    #         raise Exception("Invalid stl expression: " + str(self.expr))

    #     # Check if time step (t) is part of the lse sum. If not, return None signifying not to sum
    #     # tau = 
    #     # comma = self.expr.index(',')
    #     end = self.expr.index(']')
    #     split = end + 1
    #     # a = float(self.expr[2:comma])
    #     # b = float(self.expr[comma+1:end])
    #     # tau = len(tmdp_s)
    #     # if not (tau - 1) <= t <= b:
    #     #     return sign, None

    #     # TODO: check tmdp_s length vs horizon length
    #     # TODO: replace tau above with tau according to equation 9

    #     # remove K[.] from expr
    #     sub_expr = self.expr[split:]
    #     # Create signal as coordinate position of each state in the history
    #     if self.ts_sig_dict == None:
    #         raise Exception("State to signal mapping must be set with set_ts_sig_dict().")

    #     sig = [self.ts_sig_dict[x] for x in tmdp_s]
    #     rdeg = self.rdegree(sub_expr, sig)

    #     # add to cache dict
    #     self.rdegree_lse_cache[tmdp_s] = (sign,rdeg)

    #     return sign,rdeg

        # sig = [self.ts_sig_dict[x] for x in tmdp_s]
        # # compute the robustness degree of each ts state in tmdp state
        # ts_rdegs = [self.rdegree(sub_expr, s) for s in sig]

        # # return maximum robustness degree of ts states in tmdp state. This is effectively r(s_t^tau, phi).
        # return sign, max(ts_rdegs)

    # def rdegree(self, expr, sig):
    #     if len(sig) == 0:
    #         raise Exception('rdegree entered with empty signal!')
    #     if expr[0] in 'FG':
    #         # parse
    #         comma = expr.index(',')
    #         end = expr.index(']')
    #         a = int(expr[2:comma])
    #         b = int(expr[comma+1:end])
    #         subStl = expr[end+1:]

    #         # sig needs to be broken down for each item of the max/min according to the horizon length of the interior
    #         # If we have, for example, (F[0,2]a)&(F[0,4]b) where a and b are predicates, we could be parsing the first term which
    #         # has a horizon length of 2 and needs a signal of length 3, but we are passed a signal of length 5 because of the second
    #         # term at the same level. This could also be the case if a is another expression containing F/G. We must get the horizon length
    #         # of the interior to find out
    #         hrz_intr = self.hrz(subStl)
    #         next_sig_len = int(hrz_intr + 1)
    #         # if next_sig_len < len(sig) - b then b+ elements of sig will be unused in the part of Phi
    #         # This also defines a minimum length for sig at this point
    #         if len(sig) < b + next_sig_len:
    #             raise Exception('Signal length too short in rdegree')

    #         # max/min can take a generator. Recurse for each time step.
    #         degs = (self.rdegree(subStl, sig[t:t+next_sig_len]) for t in range(a,b+1))

    #         # max for eventually, min for globally
    #         if expr[0] == 'F':
    #             return max(degs)
    #         elif expr[0] == 'G':
    #             return min(degs)


    #     if expr[0] == '(':
    #         # has format (phi) or (phi)&(phi) or (phi)|(phi)
    #         parts = self.sep_paren(expr)
    #         if len(parts) == 1:
    #             # format was (phi). Recurse.
    #             rdeg = self.rdegree(parts[0],sig)
    #             return rdeg
    #         # TODO: check that all ops are the same for len(parts > 2)
    #         op_idx = len(parts[0]) + 2
    #         op = expr[op_idx]
    #         if op == '&':
    #             # and: minimum of two predicates
    #             # j = expr.index('&')
    #             # subStl1 = expr[1:j]
    #             # # pad second sub expr with parentheses to allow for multiple & or |
    #             # subStl2 = '(' + expr[j+1:-1] + ')'
    #             return min(self.rdegree(sub,sig) for sub in parts)
    #         elif op == '|':
    #             # or: maximum of two predicates
    #             # j = expr.index('|')
    #             # subStl1 = expr[1:j]
    #             # subStl2 = '(' + expr[j+1:-1] + ')'
    #             # return max(self.rdegree(subStl1,sig),self.rdegree(subStl2,sig))
    #             return max(self.rdegree(sub,sig) for sub in parts)
    #         else:
    #             # # predicate: remove parentheses and recurse
    #             # return self.rdegree(expr[1:-1],sig)
    #             # invalid expression
    #             raise Exception('Invalid operator in STL expression: ' + op)
    #     elif expr[0] == '!':
    #         # negation, remove the ! and return the negative
    #         subStl = expr[1:]
    #         return -1 * self.rdegree(subStl,sig)
    #     else:
    #         # This should be a simple inequality 'x<f' where f is a float
    #         f = float(expr[2:])
    #         # also sig should be length 1
    #         if len(sig) != 1:
    #             raise Exception("STL parsing reached predicate with more than one state in history")
    #         else:
    #             sig = sig[0]
    #         if 'x<' in expr:
    #             return f - sig['x']
    #         elif 'x>' in expr:
    #             return sig['x'] - f
    #         elif 'y<' in expr:
    #             return f - sig['y']
    #         elif 'y>' in expr:
    #             return sig['y'] - f
    #         else:
    #             raise Exception('Invalid stl expression: ' + str(self.expr) + ' when evaluating: ' + str(expr))

    def get_tau(self):
        return self.tau
        # # just return it if it exists
        # if self.tau != None:
        #     return self.tau
        # # PHI = K[.]phi
        # # we need hrz of phi, not PHI
        # end = self.expr.index(']')
        # phi = self.expr[end+1:]
        # # TODO assuming time step of 1
        # self.tau = int(self.hrz(phi) + 1)
        # return self.tau
        
    def get_hrz(self):
        return self.hrz
        # """
        # Recursively calculate the horizon length of expression phi. Phi is the expression passed to the constructor by default.
        # Valid formats for phi include K[.]a, (K[.]a), (K[.]a)&(K[.]b)&(...), p, and (p) where p is a predicate, K is either G or F, 
        #     and a,b are valid expressions for phi by this same definition. Additionally & is replacible with |.
        # """

        # # TODO: this is waaay over complicated compared to STL fragment in tau mdp paper

        # if phi == None:
        #     phi = self.expr
        # if 'G' not in phi and 'F' not in phi:
        #     # simple predicate
        #     return 0
        # # account for a top level conjunction/disjunction
        # # Could have  phi as 'F[.]a' or '(F[.]a)' or '(F[.]a)&(F[.]b)
        # if phi[0] == '(':
        #     # note this could be len 1
        #     sub_phis = self.sep_paren(phi)
        #     hrz = max(self.hrz(p) for p in sub_phis)
        # else:
        #     # phi is F[.]a
        #     split = phi.index(']') + 1
        #     outer = phi[:split]
        #     inner = phi[split:]
        #     comma = outer.index(',')
        #     b = float(outer[comma+1:-1])
        #     hrz = b + self.hrz(inner)
        # return int(hrz)
    
    # def sep_paren(self, phi):
    #     """
    #     Returns a list of phrases enclosed in top level parentheses
    #     """
    #     parts = []
    #     depth = 0
    #     start = 0
    #     for i,c in enumerate(phi):
    #         if c == '(':
    #             depth += 1
    #             if depth == 1:
    #                 start = i+1
    #         elif c == ')':
    #             depth -= 1
    #             if depth == 0:
    #                 parts.append(phi[start:i])
    #             elif depth < 0:
    #                 raise Exception("Mismatched parentheses in STL expression!")
    #     if depth != 0:
    #         raise Exception("Mismatched parentheses in STL expression!")
    #     return parts



