import json
import time

a_list = []
b_list = []
def check_init_value(prob_des):
    with open('data/final_sa_values.txt', 'r') as fp:
        data = json.load(fp)
          
    test_dict = {}
    for i in data:
        if eval(i)[2] == 0: 
            b_list.append(data[i][0])
            if data[i][0] < prob_des:
                a_list.append((i,data[i][0]))
    if a_list != []:
        print('<<<<<<<<<<<< assumption not hold >>>>>>>>>>>>')
        print(a_list)
        time.sleep(2)
    print(b_list)
    print(a_list)

