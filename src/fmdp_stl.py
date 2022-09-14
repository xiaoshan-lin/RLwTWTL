from pyTWTL import lomap
import itertools
import numpy as np
import networkx as nx

class FmdpState:

    def __init__(self, mdp_state, flags):
        self.mdp_s = mdp_state
        self.flags = flags


class Fmdp(lomap.Ts):

    def __init__(self, mdp, stl_expr, mdp_sig_dict):
        # O(tau*n*a)
        # with constant number of small phi, tau ~ #flags
        lomap.Ts.__init__(self, directed=True, multi=False)
        self.name = "Flag MDP"
        self.mdp = mdp
        self.stl_expr = stl_expr

        self.sig_dict = mdp_sig_dict
        self.fmdp_stl = FmdpStl(stl_expr, mdp_sig_dict)

        self.flag_max = [tau-1 for tau in self.fmdp_stl.get_tau()]

        #TODO only create states that can be accessed
        self.build_states()
        self.build_transitions()


        # set init state as mdp init state with 0 flags
        # TODO: Check that this is the correct initialization. May cause issues of mdp_init is a satisfying state.
        mdp_init = list(mdp.init.keys())[0]
        flag_init = (0.,) * self.fmdp_stl.n
        self.init = {(mdp_init, flag_init): 1}
        
    def build_states(self):
        # O(n*tau)
        mdp_states = self.mdp.g.nodes()
        flag_set = [list(range(0,m+1)) for m in self.flag_max]
        flag_product = itertools.product(*flag_set)
        self.flag_set = sorted(itertools.product(*flag_set))
        self.states = list(itertools.product(mdp_states, flag_product))
        node_attrs = [(s, self.mdp.g.nodes[s[0]]) for s in self.states]
        self.g.add_nodes_from(node_attrs)

    def build_transitions(self):
        # O(a*n*tau^{N_phi}) worst case
        
        mdp_tx_dict = nx.convert.to_dict_of_dicts(self.mdp.g)
        edge_list = []
        for fmdp_s in self.states:
            mdp_s = fmdp_s[0]
            flags = fmdp_s[1]
            edge_attr_dict = mdp_tx_dict[mdp_s]                 # dict of attrs keyed by next state
            next_mdp_s = list(edge_attr_dict.keys())                  # list of next states
            next_flags = [self.fmdp_stl.flag_update(flags, s) for s in next_mdp_s]   # next flag tuples
            # next_fmdp_s = {mf:next_mdp_s_dict[m] for mf in zip(next_mdp_s, next_flags)}
            for m,f in zip(next_mdp_s, next_flags):
                edge_list.append((fmdp_s, (m,f), edge_attr_dict[m]))
            # next_fmdp_s = {(m,f):next_mdp_s_dict[m] for m,f in zip(next_mdp_s, next_flags)}
            # self.tx_dict[fmdp_s] = next_fmdp_s

        self.g.add_edges_from(edge_list)
        # self.g.add_edge()

        # Remove states that do not have reverse neighbors
        # TODO: may be faster to selectivly create states instead
        g_rev = self.g.reverse()
        for s in list(self.g.nodes()):
            if g_rev[s] == {}:
                self.g.remove_node(s)

    def reward(self, fmdp_s, beta):
        temporal_op = self.fmdp_stl.get_outer_temporal_op()
        if temporal_op == 'F':
            r = np.exp(beta * self.fmdp_stl.sat(fmdp_s))
        elif temporal_op == 'G':
            r = -1 * np.exp(-1 * beta * self.fmdp_stl.sat(fmdp_s))
            positive_offset = np.exp(0)
            r += positive_offset
        return r

    def get_hrz(self):
        return self.fmdp_stl.get_hrz()

    def get_mdp_state(self, fmdp_s):
        return fmdp_s[0]

    def get_tau(self):
        # In the case of fmdp, this is max(tau_i)
        return int(max(self.fmdp_stl.get_tau()))

    def reset_init(self):
        pass

    def get_state_to_remove(self):
        pass

    def get_mdp(self):
        return self.mdp

    def new_ep_state(self, fmdp_s):
        # TODO: reset flag?
        return fmdp_s

    def sat(self, fmdp_s):
        return self.fmdp_stl.sat(fmdp_s)

    def is_state(self, fmdp_s):
        # TODO potentially make this more robust
        return type(fmdp_s) == tuple

    def get_null_state(self, fmdp_s):
        mdp_s = self.get_mdp_state(fmdp_s)

        # flag doesn't really matter because will be correct by t = tau-1 no matter how they start
        # but we need a consistent null state for q-table etc
        # loop through flag set until we find a state that exists
        for flags in self.flag_set:
            fmdp_s = (mdp_s, flags)
            if fmdp_s in self.g.nodes():
                break
        return fmdp_s
        

class FmdpStl:

    def __init__(self, stl_expr, mdp_sig_dict = None):
        # About O(1)
        # format
        # limited to depth of 2 {F,G}
        # Cannot be depth 1 i.e. G[0,2]x<3
        # P[t,t]p[t,t]q
        # P[t,t]p[t,t](q|q)
        # P[t,t]((p[t,t]q)&(p[t,t]q))

        # Check parenthesis
        if stl_expr.count('(') != stl_expr.count(')'):
            raise Exception("Mismatched parenthesis in STL expression!")

        self.stl_expr = stl_expr

        # extract the big phi
        end = stl_expr.index(']') + 1
        self.big_phi = stl_expr[:end]

        # extract each phi_i
        self.small_phi = stl_expr[end:]
        phi_ind = [i for i,l in enumerate(self.small_phi) if l in 'FG']
        self.n = len(phi_ind)
        phi_ind.append(len(self.small_phi))
        
        self.phi_i_list = []
        for i,j in zip(phi_ind[:-1], phi_ind[1:]):
            phi = self.small_phi[i:j].strip('()&|')
            # possibly stripped necessary ) off the end. Add some back.
            end_paren = ')' * (phi.count('(') - phi.count(')'))
            self.phi_i_list.append(phi + end_paren)

        # Get hrz (tau_i) and extract predicate of each 
        self.tau_i_list = []
        self.pred_i_list = []
        self.hrz_i_list = []
        for phi in self.phi_i_list:
            end = phi.index(']')
            hrz = float(phi[4:end])
            if hrz % 1 != 0:
                raise Exception("Non integer in STL expression. Is this allowed?")
            self.hrz_i_list.append(int(hrz))
            # TODO: Paper says not to add 1. I don't think that is correct. Check.
            self.tau_i_list.append(int(hrz) + 1)
            self.pred_i_list.append(phi[end+1:])

        self.sig_dict = mdp_sig_dict
        self.flag_max = [tau-1 for tau in self.tau_i_list]

    def get_n(self):
        return self.n

    def get_tau(self):
        return self.tau_i_list

    def get_hrz(self):
        end = self.big_phi.index(']')
        b = int(self.big_phi[4:end])
        return int(b + max(self.hrz_i_list))

    def set_mdp_sig_dict(self, mdp_sig_dict):
        self.sig_dict = mdp_sig_dict

    def get_outer_temporal_op(self):
        return self.big_phi[0]

    def flag_update(self, this_flags, next_mdp_s):
        # O(1) assuming constant # of small phi
        flags = this_flags
        sig = self.sig_dict[next_mdp_s]

        if len(flags) != self.n:
            raise Exception("Incorrect number of flags passed to flag update!")

        # loop through each flag, tau, phi and set the next flag
        next_flags = []
        for flag, fmax, phi in zip(flags, self.flag_max, self.phi_i_list):
            start = phi.index(']') + 1
            pred = phi[start:]
            satisfies = self.sat_predicate_expr(sig, pred)
            t_op = phi[0]

            if t_op == 'F' and satisfies:
                next_flags.append(fmax)
            elif t_op == 'F' and not satisfies:
                # f = max(flag - (1. / (tau - 1.)), 0.0)
                f = max(flag - 1, 0)
                next_flags.append(f)
            elif t_op == 'G' and satisfies:
                # f = min(flag + (1. / (tau - 1.)), 1.0)
                f = min(flag + 1, fmax)
                next_flags.append(f)
            elif t_op == 'G' and not satisfies:
                next_flags.append(0)
            else:
                raise Exception("Error in flag update conditions")

        # convert to integers
        # next_flags_int = [int(round(f * max_f)) for f,max_f in zip(next_flags, self.flag_max)]
        return tuple(next_flags)


    def sat(self, fmdp_s, phi = None, i=0):

        if phi == None:
            phi = self.small_phi

        if self.sig_dict == None:
            raise Exception("A dictionary mapping mdp states to a signal must be defined via set_mdp_sig_dict")

        mdp_s = fmdp_s[0]
        sig = self.sig_dict[mdp_s]

        if phi[0] in 'FG':
            end = phi.index(']') + 1
            predicate = phi[end:]
            satisfies = self.sat_predicate_expr(sig, predicate)
            flags = fmdp_s[1]
            flag_i = flags[i]

            if phi[0] == 'F':
                if flag_i > 0 or satisfies:
                    return True
                elif flag_i == 0 and not satisfies:
                    return False
                else:
                    raise Exception("Something went wrong")
            elif phi[0] == 'G':
                if flag_i == self.flag_max[i] and satisfies:
                    return True
                elif flag_i < self.flag_max[i] or not satisfies:
                    return False
                else:
                    raise Exception("Something went wrong")
            else:
                raise Exception("Something went wrong")

        elif '&' in phi or '|' in phi:
            p1, opr, p2 = self.parse_and_or(phi)
            phis_in_p1 = p1.count('G') + p1.count('F')
            sat1 = self.sat(fmdp_s, p1, i)
            sat2 = self.sat(fmdp_s, p2, i + phis_in_p1)
            if opr == '&':
                return sat1 and sat2
            elif opr == '|':
                return sat1 or sat2
            else:
                raise Exception("Invalid operator: " + opr)

        else:
            raise Exception('Reached possible predicate without temporal operator: ' + phi)
    
    def sat_predicate_expr(self, sig, pred):

        if pred[0] == '~':
            return not self.sat_predicate_expr(sig, pred[1:])

        elif '&' in pred or '|' in pred:
            p1, opr, p2 = self.parse_and_or(pred)
            sat1 = self.sat_predicate_expr(sig, p1)
            sat2 = self.sat_predicate_expr(sig, p2)
            if opr == '&':
                return sat1 and sat2
            elif opr == '|':
                return sat1 or sat2
            else:
                raise Exception("Invalid operator: " + opr)
        else:
            # This should be an 's < d'
            # TODO: allow for f(s) i.e. an expression on lhs
            if '<' in pred:
                dim, d = pred.split('<')
                return sig[dim] < float(d)
            elif '>' in pred:
                dim, d = pred.split('>')
                return sig[dim] > float(d)
            else:
                raise Exception("Invalid predicate: " + pred)


    def parse_and_or(self, phi: str) -> tuple[str]:
        """
        Returns a three tuple of two expressions seperated by an 'and' (&) or 'or' (|) 
        """

        and_or_count = phi.count('&') + phi.count('|')
        # Check for simple expression
        if and_or_count == 1:
            phi = phi.translate({ord(c): None for c in '()'}) # remove unneeded parentheses
            if '|' in phi:
                parts = phi.split('|')

                return (parts[0], '|', parts[1])
            if '&' in phi:
                parts = phi.split('&')
                return (parts[0], '&', parts[1])

        if phi[0] != '(':
            raise Exception("Invalid expression passed to and/or parser: {}. \nEach and/or must be enclosed in parentheses.".format(phi))

        parts = []
        depth = 0
        start = 0
        end = 0
        opr = ''
        for i,c in enumerate(phi):
            if c == '(':
                depth += 1
                if depth == 1:
                    start = i+1
                    if len(parts) == 1:
                        opr = phi[end+1:i]
                        if opr not in '&|':
                            raise Exception("Invalid operator: {}".format(opr))
            elif c == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    parts.append(phi[start:end])
                    if len(parts) > 2:
                        raise Exception("Each parenthetical boolean comparison can only contain two predicates.")
                elif depth < 0:
                    raise Exception("Mismatched parentheses in STL expression!")
        if depth != 0:
            raise Exception("Mismatched parentheses in STL expression!")

        if len(parts) != 2:
            if phi[0] == '(' and phi[-1] == ')':
                tup = self.parse_and_or(phi[1:-1])
                return tup
            else:
                raise Exception("invalid and/or expression: " + phi)
        return (parts[0], opr, parts[1])
