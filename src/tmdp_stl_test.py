import numpy as np
from tmdp_stl import tmdp_stl

PHI1 = 'G[0,10]F[0,3]((x>2)&(x<3)&(y>2)&(y<3))'                                             #G[0,10]F[0,3]{s10}
PHI2 = 'G[0,10](F[0,3]((x>2)&(x<3)&(y>2)&(y<3)))&(F[0,5]((x>3)&(x<4)&(y>2)&(y<3)))'         #G[0,10](F[0,3]{s10})&(F[0,5]{s14})
PHI3 = 'F[0,20]G[0,4]F[0,3]((x>2)&(x<3)&(y>2)&(y<3))'                                       #F[0,20]G[0,4]F[0,3]{s10}
PHI4 = 'F[0,20]G[0,4](F[0,3]((x>2)&(x<3)&(y>2)&(y<3)))&(F[0,5]((x>3)&(x<4)&(y>2)&(y<3)))'   #F[0,20]G[0,4](F[0,3]{s10})&(F[0,5]{s14})
PHI5 = 'G[0,20]F[0,2]((x>2)&(x<4)&(y>2)&(y<4))'


N = 4
STATE_DICT = {('s' + str(i)):((i // N) +0.5, (i % N) +0.5) for i in range(N**2)}

def lse_traj_rdegree(phi, traj, state_dict = STATE_DICT):
    beta = 4
    stl = tmdp_stl(phi,state_dict)
    tau = stl.get_tau()
    tmdp_traj = []
    reward = 0
    for t in range(tau):
        tmdp_traj.append(traj[t:len(traj) - (tau - t - 1)])
    tmdp_traj = list(zip(*tmdp_traj)) # transpose

    for tmdp_s in tmdp_traj:
        sign,rd_phi = stl.rdegree_lse(tmdp_s)
        reward += sign * np.exp(sign * beta * rd_phi)
    
    log_reward = sign * (1.0/beta) * np.log(sign * reward)
    return log_reward

def test_tmdp_states():
    stl1 = tmdp_stl(PHI1, STATE_DICT)
    print(stl1.get_tau() == 4)
    print(stl1.rdegree_lse(['s0','s5','s6','s10']))
    print(stl1.rdegree_lse(['s0','s1','s2','s3']))
    stl2 = tmdp_stl(PHI2, STATE_DICT)
    print(stl2.get_tau() == 6)
    print(stl2.hrz() == 15)
    print(stl2.rdegree_lse(['s0','s1','s4','s8','s12','s8']) == (-1, -1.5))
    print(stl2.rdegree_lse(['s0','s5','s6','s7','s11','s15']) == (-1, -0.5))
    print(stl2.rdegree_lse(['s0','s5','s9','s14','s10','s14']) == (-1, -0.5))
    print(stl2.rdegree_lse(['s0','s5','s10','s7','s11','s10']) == (-1, -0.5))
    print(stl2.rdegree_lse(['s5','s10','s14','s9','s9','s9']) == (-1, 0.5))
    print(stl2.rdegree_lse(['s5','s10','s6','s9','s13','s14']) == (-1, 0.5))
    stl3 = tmdp_stl(PHI3, STATE_DICT)
    print(stl3.get_tau() == 8)
    print(stl3.rdegree_lse(['s0','s5','s9','s10','s11','s14','s15','s11']) == (1, -0.5))
    print(stl3.rdegree_lse(['s0','s5','s9','s10','s11','s14','s15','s10']) == (1, 0.5))
    stl4 = tmdp_stl(PHI4, STATE_DICT)
    print(stl4.get_tau() == 10)
    print(stl4.rdegree_lse(['s0','s5','s9','s10','s11','s14','s15','s11','s10','s14']) == (1, -0.5))
    print(stl4.rdegree_lse(['s0','s5','s9','s10','s11','s14','s15','s10','s11','s14']) == (1, 0.5))

def test_tmdp_traj():
    traj_nums = [7, 11, 10, 15, 11, 14, 14, 11, 15, 11, 15, 15, 15, 15, 14, 13, 13, 10, 13, 10,  7,  7,  7]
    traj = ['s' + str(n) for n in traj_nums]
    print(lse_traj_rdegree(PHI5,traj))

if __name__ == '__main__':
    # test_tmdp_states()
    test_tmdp_traj()
