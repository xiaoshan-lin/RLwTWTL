
import create_environment as ce
import timeit
import yaml
import os
import copy
import networkx as nx

from pyTWTL import lomap
from pyTWTL import synthesis as synth

from tmdp_stl import Tmdp
from product_automaton import AugPa
from fmdp_stl import Fmdp
from dfa import create_dfa, save_dfa
from static_reward_mdp import StaticRewardMdp
import Q_learning as ql

CONFIG_PATH = '../configs/default_static.yaml'

NORMAL_COLOR = '\033[0m'

COLOR_DICT = {
    'pi epsilon go' : '\033[38;5;3m',   # yellow
    'explore'       : '\033[38;5;4m',   # blue
    'exploit'       : '\033[38;5;2m',   # green
    'intended'      : '\033[49m',       # no highlight
    'unintended'    : '\033[48;5;1m'    # red highlight
}

this_file_path = os.path.dirname(os.path.abspath(__file__))


def build_environment(env_cfg, twtl_cfg, mdp_type, reward_cfg, des_prob):
    """
    Create the MDP, Augmented MDP, DFA, and Augmented Product MDP 

    Parameters
    ----------
    env_cfg : dict
        Environment configuration dictionary
    twtl_cfg : dict
        TWTL constraint configuration dictionary
    mdp_type : string
        The MDP augmentation type. Either 'static rewards', 'flag-MDP', or 'tau-MDP'
    reward_cfg : dict
        Reward configuration dictionary. Expected items depends on mdp_type
    
    Returns
    -------
    AugPa
        An Augmented Product MDP
    """

    # Get values from configs
    m = env_cfg['height']
    n = env_cfg['width']
    h = 1   # depth: use only 2 dimensions
    init_state = env_cfg['init state']
    obstacles = env_cfg['obstacles']
    one_way = env_cfg['one way']
    one_way_dict = {}
    if one_way != None:
        for i in one_way:
            one_way_dict[int(i[1:])] = int(one_way[i][1:])
        print(one_way_dict)

    def xy_to_region(x,y,z):
        # x is down, y is across
        # ignore 3rd dim
        return x * n + y

    # TODO: Clean this up

    # m: length, n: width

    # =================================
    # MDP Creation
    # =================================
    ts_start_time = timeit.default_timer()
    disc = 1
    TS, obs_mat, state_mat = ce.create_ts(m,n,h)	
    path = '../data/ts_' + str(m) + 'x' + str(n) + 'x' + str(h) + '_1Ag_1.txt'
    abs_path = os.path.join(this_file_path, path)
    paths = [abs_path]
    # bases = {init_state: 'Base1'}
    bases = {}
    obs_mat = ce.update_obs_mat(obs_mat, state_mat, m, obstacles, init_state)
    TS      = ce.update_adj_mat_3D(m, n, h, TS, obs_mat)
    TS      = ce.update_one_way(TS, one_way_dict)
    ce.create_input_file(TS, state_mat, obs_mat, paths[0], bases, disc, m, n, h, 0)
    ts_file = paths
    ts_dict = lomap.Ts(directed=True, multi=False) 
    ts_dict.read_from_file(ts_file[0])
    ts = synth.expand_duration_ts(ts_dict)
    init_state_str = 'r' + str(xy_to_region(*init_state))
    ts.init = {init_state_str: 1}

    # =================================
    # Signal Creation
    # =================================
    # create dictionary mapping mdp states to a position signal for use in robustness calculation
    # apply offset so pos is middle of region
    state_to_pos = dict()
    dims = ['x', 'y', 'z']
    for s in ts.g.nodes():
        num = int(s[1:])
        #TODO: 3rd dim?
        pos = ((num // n) + 0.5, (num % n) + 0.5, 0.5)
        state_to_pos[s] = {d:p for d,p in zip(dims, pos)}
    
    ts_timecost =  timeit.default_timer() - ts_start_time


# =================================
    # DFA Creation
    # =================================
    dfa_start_time = timeit.default_timer()
    dfa_total, dfa, dfa_horizon, dfa_print_string = create_dfa(twtl_cfg, env_cfg)
    kind = twtl_cfg['DFA modification type']
    if (kind == 'total'):
        dfa = dfa_total       
    if twtl_cfg['save dfa'] == True:
        save_dfa(dfa)
        
    dfa_timecost =  timeit.default_timer() - dfa_start_time

    # =================================
    # Augmented MDP Creation
    # =================================
    aug_mdp_timer = timeit.default_timer()
    if mdp_type == 'flag-MDP':
        stl_expr = reward_cfg['STL expression']
        aug_mdp = Fmdp(ts, stl_expr, state_to_pos)
    elif mdp_type == 'tau-MDP':
        stl_expr = reward_cfg['STL expression']
        aug_mdp = Tmdp(ts, stl_expr, state_to_pos)
    elif mdp_type == 'static rewards':
        hrz = dfa_horizon
        reward_dict = reward_cfg['reward dict']
        aug_mdp = StaticRewardMdp(ts, hrz, state_to_pos, reward_dict)
    else:
        raise ValueError("invalid AUG_MDP_TYPE")
    aug_mdp_timecost = timeit.default_timer() - aug_mdp_timer
    mdp_horizon = aug_mdp.get_hrz()

    if mdp_horizon != dfa_horizon:
        if mdp_type == 'static rewards':
            raise RuntimeError(f'Static rewards MDP has an incorrect time horizon. This is likely \
                an implementation error. MDP horizon: {mdp_horizon}, TWTL horizon: {dfa_horizon}')
        else:
            raise ValueError(f'STL and TWTL time horizon mismatch. Please adjust either spec \
                so the horizons match. STL time horizon: {mdp_horizon}, TWTL time horizon: {dfa_horizon}')


    # =================================
    # Augmented Product MDP Creation
    # =================================
    pa_start_time = timeit.default_timer()
    critical_time = twtl_cfg['critical_time']
    pa_or = AugPa(aug_mdp, dfa, dfa_horizon, n, m, kind, critical_time, des_prob)
    pa = copy.deepcopy(pa_or)	      # copy the pa
    pa_timecost =  timeit.default_timer() - pa_start_time

    # Compute the energy of the states
    energy_time = timeit.default_timer()
    # pa.compute_energy()
    energy_timecost =  timeit.default_timer() - energy_time

    init_state_num = init_state[0] * n + init_state[1]

    # =================================
    # Print information
    # =================================
    print('##### PICK-UP and DELIVERY MISSION #####' + "\n")
    print('Initial Location  : ' + str(init_state) + ' <---> Region ' + str(init_state_num))
    print(dfa_print_string)
    print('State Matrix : ')
    print(state_mat)
    print("\n")
    print('Mission Duration  : ' + str(dfa_horizon) + ' time steps')
    print('Time PA state size: {}\n'.format(pa.get_tpa_state_size()))
    print('Time Cost:')
    print('TS creation time (s):            {:<7}'.format(ts_timecost))
    print('Augmented MDP creation time (s): {:<7}'.format(aug_mdp_timecost))
    print('DFA creation time (s):           {:<7}'.format(dfa_timecost))
    print('PA creation time (s):            {:<7}'.format(pa_timecost))
    print('PA energy calculation time (s):  {:<7}'.format(energy_timecost))

    return pa


def main():
    """
    Main function. Read in configuration values, construct the Pruned Time-Product MDP, 
    find the optimal policy, and test the optimal policy.
    """

    # Load default config
    my_path = os.path.dirname(os.path.abspath(__file__))
    def_cfg_rel_path = CONFIG_PATH
    def_cfg_path = os.path.join(my_path, def_cfg_rel_path)
    with open(def_cfg_path, 'r') as f:
        config = yaml.safe_load(f)

    # stl_expr = 'G[0,10]F[0,3](((x>1)&(x<2))&((y>3)&(y<4)))'

    # ==== Read in configuration values ====
    # Q-learning config
    qlearn_cfg = config['Q-learning config']
    num_episodes = qlearn_cfg['number of episodes']     # of episodes
    learn_rate = qlearn_cfg['learning rate']
    discount   = qlearn_cfg['discount']

    explore_prob_start = qlearn_cfg['explore probability start']
    explore_prob_end = qlearn_cfg['explore probability end']
    # start * decay^(num_eps - 1) = end
    explore_prob_decay = (explore_prob_end/explore_prob_start)**(1/(num_episodes-1))

    # environment config
    env_cfg = config['environment']
    eps_unc    = env_cfg['real action uncertainty'] # Uncertainity in actions, real uncertainnity in MDP
    eps_unc_learning = env_cfg['over estimated action uncertainty'] # Overestimated uncertainity used in learning

    # TWTL constraint config
    twtl_cfg = config['TWTL constraint']
    des_prob = twtl_cfg['desired satisfaction probability'] # Minimum desired probability of satisfaction

    mdp_type = config['MDP type']
    if mdp_type == 'static rewards':
        reward_cfg = config['static rewards']
    else:
        reward_cfg = config['aug-MDP rewards']

    test_cfg = config['test_policy config']
    num_episodes_test = test_cfg['number of episodes']
    use_saved_policy =  test_cfg['use saved policy']
    policy_file =  test_cfg['policy_file']
    policy_pic_file =  test_cfg['policy_pic_file']

    # ==== Construct the Pruned Time-Product MDP ====
    prep_start_time = timeit.default_timer()

    # Construct the Product MDP
    pa = build_environment(env_cfg, twtl_cfg, mdp_type, reward_cfg, des_prob)
    # Prune it at each time step
    prune_start = timeit.default_timer()
    print("____start pruning____")
    pa.prune_actions(eps_unc_learning, des_prob)
    prune_end = gen_start = timeit.default_timer()
    print('Time PA action pruning time (s): {}'.format(prune_end - prune_start))
    pa.gen_new_ep_states()
    gen_end = timeit.default_timer()

    prep_end_time = timeit.default_timer()

    print('New ep/traj generation time (s): {}'.format(gen_end - gen_start))
    print('')
    print('Total environment creation time: {}'.format(prep_end_time - prep_start_time))
    print('')

    # ==== Find the optimal policy ====
    print('learning with {} episodes'.format(num_episodes))
    timer = timeit.default_timer()
    if not use_saved_policy:
        pi, pi_c = ql.Q_learning(pa, num_episodes, eps_unc, learn_rate, discount, explore_prob_decay, explore_prob_start)
    else:
        pi = None
        pi_c = None
    qlearning_time = timeit.default_timer() - timer
    print('learning time: {} seconds'.format(qlearning_time))

    # ==== test policy ====
    stl_expr = config['aug-MDP rewards']['STL expression']

    ql.test_policy(pi, pi_c, pa, stl_expr, eps_unc, num_episodes_test, mdp_type, use_saved_policy, policy_file, policy_pic_file)




if __name__ == '__main__':
    main()

