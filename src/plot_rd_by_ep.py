import matplotlib.pyplot as plt
import numpy as np

from plot_route import ansi_str_to_path_int
from STL import STL

STL_EXPR = 'G[0,10]F[0,3](((x>1)&(x<2))&((y>3)&(y<4)))'
HEIGHT = 2
WIDTH = 6

def plot_rd_by_ep(file_, cut_at = None):
    with open(file_, 'r') as f:
        lines = f.readlines()

    if cut_at != None:
        lines = lines[:cut_at]

    trajs = [ansi_str_to_path_int(l) for l in lines]
    sigs = [states_to_sig(r) for r in trajs]

    parser = STL(STL_EXPR)
    rds = [parser.rdegree(s) for s in sigs]
    
    plt.plot(rds, '.', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Robustness Degree')
    # plt.title('Robustness Degree of each Episode while Learning')
    plt.show()

def states_to_sig(state_num):
    x, y = np.unravel_index(state_num, (HEIGHT, WIDTH)) # pylint: disable=unbalanced-tuple-unpacking
    x_mid = x + 0.5
    y_mid = y +0.5
    sig = [dict(x=a, y=b) for a,b in zip(x_mid,y_mid)]
    return sig


if __name__ == '__main__':
    file_path = '../output/case_study_1_train_flag.txt'
    plot_rd_by_ep(file_path, cut_at=1000)
