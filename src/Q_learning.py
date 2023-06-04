import os, time
import numpy as np
import random
from tqdm import tqdm
import json
from numpy import remainder as rem

from STL import STL

NORMAL_COLOR = '\033[0m'
COLOR_DICT = {
    'pi epsilon go' : '\033[38;5;3m',   # yellow
    'explore'       : '\033[38;5;4m',   # blue
    'exploit'       : '\033[38;5;2m',   # green
    'intended'      : '\033[49m',       # no highlight
    'unintended'    : '\033[48;5;1m'    # red highlight
}


this_file_path = os.path.dirname(os.path.abspath(__file__))

def Q_learning(pa, episodes, eps_unc, learn_rate, discount, eps_decay, epsilon, log=True):
    """
    Find the optimal policy using Q-learning

    Parameters
    ----------
    pa : AugPa
        The Augmented Product MDP
    episodes : int
        The number of episodes of learning
    eps_unc : float
        The real action uncertainty. This is the probability of an unintended transition.
    learn_rate : float
        The learning rate used in the Q update function
    discount : float
        The future value discount used in the Q update function
    eps_decay : float
        The decay rate of the exploration probability. 
        (inital prob) * decay^(episodes - 1) = (final prob)
    epsilon : float
        The initial exploration probability
    log : boolean
        Whether log files should be created
        default is True

    Returns
    -------
    dict
        The optimal policy pi as a dict of dicts. The outer is keyed by the time step
        and the inner is keyed by the Product MDP state. The value is the optimal next Product MDP state.
    """

    # Log state sequence and reward
    trajectory_reward_log = []
    mdp_traj_log = []
    tr_log_file = os.path.join(this_file_path, '../output/trajectory_reward_log.txt')
    mdp_log_file = os.path.join(this_file_path, '../output/mdp_trajectory_log.txt')
    log = True

    # truncate file
    open(tr_log_file, 'w').close()
    open(mdp_log_file, 'w').close()

    # Count trajectories that reach an accepting state (pass the TWTL task)
    twtl_pass_count = 0
    ep_rew_sum = 0
    ep_rewards = np.zeros(episodes)

    # initial state,time
    z,t_init,init_traj = pa.initial_state_and_time()

    # initialize Q table
    init_val = 0
    time_steps = pa.get_hrz()
    qtable = {t:{} for t in range(t_init, time_steps+1)}

    for t in qtable:
        qtable[t] = {q[:2]:{} for q in pa.newg.nodes() if q[:2]+(t,) in pa.newg.nodes()}
        for p in qtable[t]:
            if pa.pruned_actions[t][p] != []:
                qtable[t][p] = {a:init_val + np.random.normal(0,0.0001) for a in pa.pruned_actions[t][p]}
            else:
                qtable[t][p] = {a:init_val + np.random.normal(0,0.0001) for a in pa.pi_c[t][p]}

    # initialize optimal policy pi on pruned time product automaton
    pi = {t:{} for t in qtable}
    pi_c = {t:{} for t in qtable}
    for t in pi:
        for p in pa.pruned_actions[t]:
            if pa.pruned_actions[t][p] != []:
                pi[t][p] = max(pa.pruned_actions[t][p], key=qtable[t][p].get)
            if pa.pi_c[t][p] != "true":
                pi_c[t][p] = max(pa.pi_c[t][p], key=qtable[t][p].get)
  
    # Make an entry in q table for learning initial states and initialize pi
    if pa.is_STL_objective:
        qtable[0] = {p:{} for p in pa.get_null_states()}
        pi[0] = {}
        for p in qtable[0]:
            qtable[0][p] = {q:init_val + np.random.normal(0,0.0001) for q in pa.get_new_ep_states(p)}
            pi[0][p] = max(qtable[0][p], key=qtable[0][p].get)

    if log:
        trajectory_reward_log.extend(init_traj)
        init_mdp_traj = [pa.get_mdp_state(z) for z in init_traj]
        mdp_traj_str = ''
        mdp_traj_str += COLOR_DICT['explore'] + '{:<4}'.format('0:')
        for x in init_mdp_traj:
            mdp_traj_str += '{:<4}'.format(x)

    #log_range = [(0,50), (39900,40000),(149900,150000)]
    log_range = []
    save_ep=[]
    for i in log_range:
        save_ep.extend(list(range(i[0],i[1])))

    # Loop for number of training episodes
    pi_c_trigger = False
    twtl_pass_count_list = []
    ave_reward = 0

    for ep in tqdm(range(episodes)):
        for t in range(t_init, time_steps+1):        
            if pa.is_accepting_state(z) or z[1] == 'trash' or pa.opt_s_value[z+(t,)]==0 or t in pa.critical_time:
                pi_c_trigger = False

            # pruned_actions
            if t < time_steps:
                pruned_actions = pa.pruned_actions[t][z]
            else:
                pruned_actions = [pa.states_to_action(z, neighbor) for neighbor in pa.g.neighbors(z)]

            if pi_c_trigger or pruned_actions == []:
                pi_c_trigger = True
                if np.random.uniform() < epsilon:   # Explore
                    action_chosen = random.choice(pa.pi_c[t][z])
                else:                               # Exploit
                    action_chosen = pi_c[t][z]
                action_chosen_by = 'pi epsilon go'
            else:
                if np.random.uniform() < epsilon:   # Explore
                    action_chosen = random.choice(pruned_actions)
                    action_chosen_by = "exploit"
                else:                               # Exploit
                    action_chosen = pi[t][z]
                    action_chosen_by = "exploit"

            # Take the action, result may depend on uncertainty
            next_z = pa.take_action(z, action_chosen, eps_unc)
            if pa.states_to_action(z, next_z) == action_chosen:
                action_result = 'intended'
            else:
                action_result = 'unintended'

            reward = pa.reward(next_z)

            # Update q value
            cur_q = qtable[t][z][action_chosen]
            if next_z in qtable[(t+1)%(time_steps+1)]:
                future_qs = qtable[(t+1)%(time_steps+1)][next_z]
                max_future_q = max(future_qs.values())
            else:
                future_qs = qtable[0][pa.get_null_state(next_z)]
                max_future_q = max(future_qs.values())                            
            new_q = (1 - learn_rate) * cur_q + learn_rate * (reward + discount * max_future_q)
            qtable[t][z][action_chosen] = new_q

            # Update optimal policy
            if action_chosen_by == 'pi epsilon go':
                pi_c[t][z] = max(pa.pi_c[t][z], key=qtable[t][z].get)
            elif action_chosen_by == 'exploit':
                pi[t][z] = max(pruned_actions, key=qtable[t][z].get)    

            # track sum of rewards
            ep_rew_sum += reward

            if log:
                trajectory_reward_log.append(next_z)
                mdp_str = COLOR_DICT[action_result] + COLOR_DICT[action_chosen_by] + '{:<4}'.format(pa.get_mdp_state(next_z))
                mdp_traj_str += mdp_str
            
            z = next_z
            if t == time_steps - 1:
                final_z = z

        pi_c_trigger = False
        epsilon = epsilon * eps_decay

        if pa.is_accepting_state(final_z):
            twtl_pass_count_list.append(1)
            twtl_pass_count += 1
        else:
            twtl_pass_count_list.append(0)

        ep_rewards[ep] = ep_rew_sum
        ave_reward += ep_rew_sum
        if rem(ep,1000)==0:
            print('average_reward:{}'.format(ave_reward/1000))
            ave_reward = 0
        ep_rew_sum = 0

        z = pa.get_null_state(z)
        init_traj = [z]
        if pa.is_STL_objective:
            # TODO
            #FIXME: pi[0][z] could be None
            # Choose init state either randomly or by pi
            if np.random.uniform() < epsilon:   # Explore
                possible_init_zs = list(qtable[0][z].keys())
                init_z = random.choice(possible_init_zs)
                action_chosen_by = "exploit"
            else:                               # Exploit
                init_z = pi[0][z]
                action_chosen_by = "exploit"

            init_traj = pa.get_new_ep_trajectory(z, init_z)

            # Update qtable and optimal policy
            reward = pa.reward(init_z)
            cur_q = qtable[0][z][init_z]
            future_qs = qtable[t_init][init_z]
            max_future_q = max(future_qs.values())
            new_q = (1 - learn_rate) * cur_q + learn_rate * (reward + discount * max_future_q)
            qtable[0][z][init_z] = new_q
            pi[0][z] = max(qtable[0][z], key=qtable[0][z].get)

        else:
            init_z = z
        z = init_z

        if log:
            if ep in save_ep:
                mdp_traj_log.append(mdp_traj_str)
            mdp_traj_str = ''
            mdp_traj_str += COLOR_DICT['explore'] + '{:<4}'.format(str(ep+1)+':')
            for pa_s in init_traj:
                mdp_s = pa.get_mdp_state(pa_s)
                mdp_traj_str += '{:<4}'.format(mdp_s)

            with open(mdp_log_file, 'a') as log_file:
                for line in mdp_traj_log:
                    log_file.write(line)
                    log_file.write('\n')
        
    if log:
        with open(tr_log_file, 'a') as log_file:
            log_file.write(str(trajectory_reward_log))
            log_file.write('\n')

        trajectory_reward_log = init_traj[:]
        init_mdp_traj = [pa.get_mdp_state(p) for p in init_traj]
        
    print("TWTL mission success: {} / {} = {}".format(twtl_pass_count, episodes, twtl_pass_count/episodes))   

    with open('data/training_sat_count.txt', 'w') as fp:
        json.dump(twtl_pass_count_list, fp)

    np.save('data/ep_rewards.npy', ep_rewards)
    save_policy(pi, 'data/learned_policy.json')
    save_policy(pi_c, 'data/learned_pic_policy.json')

    return pi, pi_c

def save_policy(pi, file_name):
    test_pi = {}
    for t in pi:
         test_pi[t] = {str(z):pi[t][z] for z in pi[t]}
    with open(file_name,'w') as fp:
        json.dump(test_pi, fp)

def test_policy(pi, pi_c, pa, stl_expr, eps_unc, iters, mdp_type, use_saved_policy, policy_file, policy_pic_file):
    """
    Test a policy for a certian number of episodes and print 
        * The constraint mission success rate, 
        * The average sum of rewards for each episode
        * The objective (STL) mission success rate (if applicable)
        * The average robustness degree of each episode (if applicable)
    
    Parameters
    ----------
    pi : dict
        A policy as a dict of dicts. The outer is keyed by the time step
        and the inner is keyed by the Product MDP state. The value is an adjacent Product MDP state.
    pa : AugPA
        The Augmented Product MDP
    stl_expr : string
        The STL expression that represents the objective TODO: make this optional for the case of static rewards
    eps_unc : float
        The real action uncertainty. This is the probability of an unintended transition.
    iters : int
        The number of episodes to test over
    mdp_type : string
        The MDP augmentation type. Either 'static rewards', 'flag-MDP', or 'tau-MDP'
    Returns('r14', 4)
    -------
    float
        The STL satisfaction rate TODO: return this only when applicable
    float
        The TWTL satisfaction rate
    
    """    
    if use_saved_policy:
        if policy_file == None:
           policy_file = 'data/learned_policy.json'
           policy_pic_file = 'data/learned_pic_policy.json'
        with open(policy_file,'r') as fp:
            data = json.load(fp)
            pi = {}
            for t in data:
                pi[eval(t)] = {eval(i):data[t][i] for i in data[t]}

        with open(policy_pic_file,'r') as fp:
            data = json.load(fp)
            pi_c = {}
            for t in data:
                pi_c[eval(t)] = {eval(i):data[t][i] for i in data[t]}

    print('Testing optimal policy with {} episodes'.format(iters))

    mdp_log_file = os.path.join(this_file_path, '../output/test_policy_trajectory_log.txt')
    open(mdp_log_file, 'w').close() # clear file
    pas_log_file = os.path.join(this_file_path, '../output/pas_test_policy_trajectory_log.txt')
    open(pas_log_file, 'w').close() # clear file
    log = True

    z,t_init,init_traj = pa.initial_state_and_time()
    time_steps = pa.get_hrz()

    if log:
        mdp_traj_str = ''
        mdp_traj_str += COLOR_DICT['explore'] + '{:<4}'.format('1:')
        pas_traj_str = ''
        pas_traj_str += COLOR_DICT['explore'] + '{:<4}'.format('1:')
        mdp_traj_log = []
        pas_traj_log = []
        init_mdp_traj = [pa.get_mdp_state(z) for z in init_traj]
        init_pas_traj = [z for z in init_traj]
        for x in init_mdp_traj:
            mdp_traj_str += NORMAL_COLOR + '{:<4}'.format(x)
        for x in init_pas_traj:
            pas_traj_str += NORMAL_COLOR + '{:<4}'.format(str(x))
    
    mdp_traj = [pa.get_mdp_state(z) for z in init_traj]
    pas_traj = [z for z in init_traj]

    # count TWTL satsifactions
    twtl_pass_count = 0

    # Count STL satisfactions and avg robustness
    if pa.is_STL_objective:
        parser = STL(stl_expr)
        stl_sat_count = 0
        stl_rdeg_sum = 0
    
    # count sum of rewards
    reward_sum = 0
    pi_c_trigger = False
    
    for idx in range(iters):
        for t in range(t_init, time_steps+1):
            if pa.is_accepting_state(z) or z[1] == 'trash' or pa.opt_s_value[z+(t,)]==0 or t in pa.critical_time:
                pi_c_trigger = False

            reward_sum += pa.reward(z)  
            pruned_actions = pa.pruned_actions[t][z]

            if pi_c_trigger or pruned_actions == []:
                pi_c_trigger = True
                action_chosen = pi_c[t][z]
                action_chosen_by = 'pi epsilon go'
            else:
                action_chosen_by = 'exploit'
                action_chosen = pi[t][z]           

            # take action
            next_z = pa.take_action(z, action_chosen, eps_unc)
            action_result = 'intended' if pa.states_to_action(z, next_z) == action_chosen else 'unintended'
             
            if log:
                if t < time_steps:
                    mdp_str = COLOR_DICT[action_result] + COLOR_DICT[action_chosen_by] + '{:<4}'.format(pa.get_mdp_state(next_z))
                    mdp_traj_str += mdp_str

                    pas_str = COLOR_DICT[action_result] + COLOR_DICT[action_chosen_by] + '{:<4}'.format(str(next_z))
                    pas_traj_str += pas_str

            z = next_z
            if t == time_steps - 1:
                final_z = z

            mdp_traj.append(pa.get_mdp_state(next_z))
            pas_traj.append(next_z)

        pi_c_trigger = False

        if pa.is_accepting_state(final_z):
            twtl_pass_count += 1

        z_null = pa.get_null_state(z)
        if pa.is_STL_objective:          
            z_init = pi[0][z_null]
            init_traj = pa.get_new_ep_trajectory(z,z_init)
            mdp_sig = [pa.aug_mdp.sig_dict[x] for x in mdp_traj]
            rdeg = parser.rdegree(mdp_sig)
            if rdeg > 0:
                stl_sat_count += 1
            stl_rdeg_sum += rdeg
        else:
            z_init = z_null
            init_traj = [z_init]

        z = z_init

        mdp_traj = [pa.get_mdp_state(p) for p in init_traj]
        pas_traj = [p for p in init_traj]

        if log:
            if pa.is_STL_objective:
                mdp_traj_str += NORMAL_COLOR + '| {:>6}'.format(rdeg)
            else:
                mdp_traj_str += NORMAL_COLOR + '| {:>6}'.format(pa.is_accepting_state(final_z))
            mdp_traj_log.append(mdp_traj_str)
            mdp_traj_str = ''
            mdp_traj_str += COLOR_DICT['explore'] + '{:<4}'.format(str(idx+2)+':')

            for pa_s in init_traj:
                mdp_s = pa.get_mdp_state(pa_s)
                mdp_traj_str += NORMAL_COLOR + '{:<4}'.format(mdp_s)
            
            if pa.is_STL_objective:
                pas_traj_str += NORMAL_COLOR + '| {:>6}'.format(rdeg)
            else:
                pas_traj_str += NORMAL_COLOR + '| {:>6}'.format(pa.is_accepting_state(final_z))
            
            pas_traj_log.append(pas_traj_str)
            pas_traj_str = ''
            pas_traj_str += COLOR_DICT['explore'] + '{:<4}'.format(str(idx+2)+':')
            for pa_s in init_traj:
                pas_traj_str += NORMAL_COLOR + '{:<4}'.format(str(pa_s))

    if log:
        with open(mdp_log_file, 'a') as log_file:
            for line in mdp_traj_log:
                log_file.write(line)
                log_file.write('\n')

        with open(pas_log_file, 'a') as log_file:
            for line in pas_traj_log:
                log_file.write(line)
                log_file.write('\n')

    twtl_sat_rate = twtl_pass_count/iters
    
    print("TWTL mission success: {} / {} = {}".format(twtl_pass_count, iters, twtl_pass_count/iters))
    print("Avg episode sum of rewards: {}".format(reward_sum/iters))
    if mdp_type != 'static rewards':
        stl_sat_rate = stl_sat_count/iters
        print("STL mission success: {} / {} = {}".format(stl_sat_count, iters, stl_sat_count/iters))
        print("Avg robustness degree: {}".format(stl_rdeg_sum/iters))

