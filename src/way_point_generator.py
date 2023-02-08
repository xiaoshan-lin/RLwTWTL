file1 = open('data/demo.txt', 'r')
import csv
Lines = file1.readlines()
a = []
for l in Lines:
    if list(l.strip().split(" ")) != ['']:
        b = list(l.strip().split(" "))
        c = [int(i[1:]) for i in b if i!='']
        a.append(c)

del a[1]
print(a)

sx = 5
sy = 5
l = 0.3

name = ['10','1000','10000']
header = ['id', 'x[m]', 'y[m]', 'z[m]', 't[s]']
d_id = 7
dt = 2
for idx,i in enumerate(name):
    result = a[idx]
    print('----------------')
    with open('way_point_{}.csv'.format(i), 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)
        t = 0
        for d in result:
            x_idx = d//sy
            y_idx = d%sy
            print(d,x_idx,y_idx)
            x_val = l/2 + l*x_idx
            y_val = l/2 + l*y_idx
            x_val = x_val - l*2
            y_val = y_val - l*3
            data = [d_id,x_val,y_val,0.4,t]
            
            t = t + dt
            writer.writerow(data)








