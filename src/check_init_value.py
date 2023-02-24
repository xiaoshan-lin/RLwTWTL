import json
import time, os, yaml

a_list = []
b_list = []
CONFIG_PATH = '../configs/default_static.yaml'
my_path = os.path.dirname(os.path.abspath(__file__))
def_cfg_rel_path = CONFIG_PATH
def_cfg_path = os.path.join(my_path, def_cfg_rel_path)
with open(def_cfg_path, 'r') as f:
    config = yaml.safe_load(f)
twtl_cfg = config['TWTL constraint']
critical_time = twtl_cfg['critical_time']
def check_init_value(des_prob):
    with open('data/final_sa_values.txt', 'r') as fp:
        data = json.load(fp)         
    test_dict = {}
    for i in data:
        if eval(i)[2] in critical_time and eval(i)[1] != "trash": 
            b_list.append(data[i][0])
            if data[i][0] < des_prob**(1/len(critical_time)):
                a_list.append((i,data[i][0]))
    if a_list != []:
        print('<<<<<<<<<<<< assumption not hold >>>>>>>>>>>>')
        print(a_list)
        time.sleep(2)
    else:
        print('<<<<<<<<<<<< assumption holds >>>>>>>>>>>>')
    print(b_list)
    print(a_list)

