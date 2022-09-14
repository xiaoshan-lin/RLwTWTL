import os
from unittest import result
import yaml
import timeit
import matplotlib.pyplot as plt
import numpy as np

from main import build_environment
import Q_learning as ql

CONFIG_PATH = '../configs/default_ECC.yaml'
DATA_PATH = '../output/stl_sat_rate_by_MDP_type_and_episodes.yaml'

NUM_EPS_LIST = [500, 1000, 3000, 5000, 10000, 50000]
# NUM_EPS_LIST = [500]
TEST_RUNS = 1000
# TEST_RUNS = 100
LEARNING_RUNS = 20
# LEARNING_RUNS = 2

this_file_path = os.path.dirname(os.path.abspath(__file__))

def compare_sat_rate_by_num_eps(num_eps_list: list[int], test_runs: int, learning_runs: int):

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
    # num_episodes = qlearn_cfg['number of episodes']     # of episodes
    learn_rate = qlearn_cfg['learning rate']
    discount   = qlearn_cfg['discount']

    explore_prob_start = qlearn_cfg['explore probability start']
    explore_prob_end = qlearn_cfg['explore probability end']
    # start * decay^(num_eps - 1) = end

    # environment config
    env_cfg = config['environment']
    eps_unc    = env_cfg['real action uncertainty'] # Uncertainity in actions, real uncertainnity in MDP
    eps_unc_learning = env_cfg['over estimated action uncertainty'] # Overestimated uncertainity used in learning

    # TWTL constraint config
    twtl_cfg = config['TWTL constraint']
    des_prob = twtl_cfg['desired satisfaction probability'] # Minimum desired probability of satisfaction

    reward_cfg = config['aug-MDP rewards']

    mdp_types = ['flag-MDP', 'tau-MDP']
    sat_rate_dict = {mt:{ne:[] for ne in num_eps_list} for mt in mdp_types}
    twtl_rate_dict = {mt:{ne:[] for ne in num_eps_list} for mt in mdp_types}

    for mdp_type in mdp_types:

        # ==== Construct the Pruned Time-Product MDP ====
        prep_start_time = timeit.default_timer()

        # Construct the Product MDP
        pa = build_environment(env_cfg, twtl_cfg, mdp_type, reward_cfg)
        # Prune it at each time step
        prune_start = timeit.default_timer()
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


        for num_eps in num_eps_list:
            explore_prob_decay = (explore_prob_end/explore_prob_start)**(1/(num_eps-1))

            for _ in range(learning_runs):

                # Do learning and get satisfaction rate
                # ==== Find the optimal policy ====
                print('learning with {} episodes'.format(num_eps))
                timer = timeit.default_timer()
                pi = ql.Q_learning(pa, num_eps, eps_unc, learn_rate, discount, explore_prob_decay, explore_prob_start)
                qlearning_time = timeit.default_timer() - timer
                print('learning time: {} seconds'.format(qlearning_time))

                # ==== test policy ====
                stl_expr = config['aug-MDP rewards']['STL expression']
                stl_sat_rate, twtl_sat_rate = ql.test_policy(pi, pa, stl_expr, eps_unc, test_runs, mdp_type)
                sat_rate_dict[mdp_type][num_eps].append(stl_sat_rate)
                twtl_rate_dict[mdp_type][num_eps].append(twtl_sat_rate)

    data_file = os.path.join(this_file_path, DATA_PATH)

    with open(data_file, 'w') as file:

        yaml.dump({'stl sat rate': sat_rate_dict, 'twtl sat rate':twtl_rate_dict}, file)


def plot_results():

    data_file = os.path.join(this_file_path, DATA_PATH)
    with open(data_file) as file:
        results_dict = yaml.load(file, Loader=yaml.Loader)

    fig, ax = plt.subplots(2,1, sharex=True, sharey=True)
    ax1, ax2 = ax

    def plot(ax: plt.Axes, name: str):
        sat_rate_by_eps = results_dict['stl sat rate'][name]
        # ax.boxplot(sat_rate_by_eps.values(), sat_rate_by_eps.keys()))
        sat_rate_arr = np.array(list(sat_rate_by_eps.values()))
        means = np.mean(sat_rate_arr, axis=1)
        print(name, means)
        # stds = np.std(sat_rate_arr, axis=1)
        low = np.min(sat_rate_arr, axis=1)
        high = np.max(sat_rate_arr, axis=1)
        x = list(sat_rate_by_eps.keys())
        ax.plot(high, 'g--', label='Maximum')
        ax.plot(means, 'k', label='Mean')
        ax.plot(low, 'r--', label='Minimum')
        ax.set_xticks(list(range(6)))
        ax.set_xticklabels(list(sat_rate_by_eps.keys()))
        for i in range(len(x)):
            ax.text(i, means[i]+0.15, f'{means[i]:.2}', size=10)

    plot(ax1, 'flag-MDP')
    plot(ax2, 'tau-MDP')
    ax1.set_ylabel('F-MDP')
    ax2.set_ylabel('tau-MDP')
    ax2.set_xlabel('# of Training Episodes')
    # ax2.text(0.5, 0.5, 0.5)
    fig.suptitle('STL Satisfaction Rate by Number of Episodes')
    # fig.text(0.04, 0.5, 'STL Satisfaction Rate', va='center', rotation='vertical')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # compare_sat_rate_by_num_eps(NUM_EPS_LIST, TEST_RUNS, LEARNING_RUNS)
    plot_results()