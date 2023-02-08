import json

with open('data/prune_actions.txt', 'r') as fp:
            data_05 = json.load(fp)

with open('data/acc_result/case_3/e01/p02/new/prune_actions.txt', 'r') as fp:
            data_02 = json.load(fp)

equal = 0
more = 0
less = 0
empty = 0

for t in data_02:
    for p in data_02[t]:
        if len(data_02[t][p]) == len(data_05[t][p]):
            equal+=1
        elif len(data_02[t][p]) > len(data_05[t][p]):
            more+=1
        else:
            less+=1

        if len(data_02[t][p]) > 0 and len(data_05[t][p])==0:
            empty +=1
print('equal={}'.format(equal))
print('more={}'.format(more))
print('less={}'.format(less))
print('empty={}'.format(empty))
