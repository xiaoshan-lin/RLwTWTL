import json

with open('data/time_traj_log.txt','r') as fp:
    data = json.load(fp)

a = [{},{},{}]

for i in data:
    for idx, j in enumerate(i):
        if idx == 0:
            if j not in a[idx]:
                a[idx][j] = 1
            else:
                a[idx][j] += 1
        else:
            if j-i[idx-1] not in a[idx]:
                a[idx][j-i[idx-1]] = 1
            else:
                a[idx][j-i[idx-1]] += 1

print(a)
with open('data/counting_result.json','w') as fp:
    json.dump(a,fp)

    
