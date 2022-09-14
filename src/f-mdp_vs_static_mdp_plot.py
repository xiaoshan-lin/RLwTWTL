import re
import matplotlib.pyplot as plt

FILE1 = '../output/plot_file1.txt'
FILE2 = '../output/plot_file2.txt'
LINE1 = 6
LINE2 = 5
NAME1 = 'Static Reward MDP'
NAME2 = 'Flag MDP'
# 'G[0,24]((F[0,6]((x>3)&((y>1)&(y<2))))&(F[0,6](((x>1)&(x<2))&((y>2)&(y<3)))))'
STL = 'G[0,24] ((F[0,6] r13) & (F[0,6] r6))'

def get_traj_from_file(file_path, line):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    line = lines[line - 1]
    print(line)
    line = re.sub(r'\033\[(\d|;)+?m', '', line)
    traj = line.split()
    return traj

print('Colored trajectories:')
mdp_traj_1 = get_traj_from_file(FILE1, LINE1)
mdp_traj_2 = get_traj_from_file(FILE2, LINE2)

if len(mdp_traj_1) != len(mdp_traj_2):
    raise Exception('Trajectories of different lengths')

n = len(mdp_traj_1)

# remove 'r' and convert to int
int_traj_1 = [int(s[1:]) for s in mdp_traj_1]
int_traj_2 = [int(s[1:]) for s in mdp_traj_2]
t = list(range(n))

fig, ax = plt.subplots()
ax.plot(t,int_traj_1,label=NAME1, color='green')
ax.plot(t,int_traj_2,label=NAME2, color='lightblue')
reward1 = [13] * n
reward2 = [6] * n
ax.plot(t,reward1,reward2, color='red', linestyle='dashed', linewidth=0.85)
ax.legend()
ax.set_title(STL)
plt.show()
