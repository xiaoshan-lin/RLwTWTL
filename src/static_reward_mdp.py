from pyTWTL import lomap

class StaticRewardMdp(lomap.Ts):

    def __init__(self, mdp, hrz, mdp_sig_dict, mdp_reward_dict):
        lomap.Ts.__init__(self, directed=True, multi=False)
        self.name = "Static Reward MDP"
        self.mdp = mdp
        self.rew_dict = mdp_reward_dict
        self.sig_dict = mdp_sig_dict
        self.hrz = hrz
        
        self.init = mdp.init
        self.g = mdp.g

    def reward(self, mdp_s, beta = None):
        try:
            r = self.rew_dict[mdp_s]
        except(KeyError):
            return 0
        return r

    def get_hrz(self):
        return self.hrz

    def get_mdp_state(self, mdp_s):
        return mdp_s

    def get_tau(self):
        return 1

    def reset_init(self):
        pass

    def get_state_to_remove(self):
        pass

    def get_mdp(self):
        return self.mdp

    def new_ep_state(self, mdp_s):
        return mdp_s

    def sat(self, mdp_s):
        s = (self.reward(mdp_s) > 0)
        return s

    def is_state(self, mdp_s):
        exists = (mdp_s in self.g.nodes())
        return exists

    def get_null_state(self, mdp_s):
        return mdp_s