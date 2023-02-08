import json
  
# reading the data from the file
with open('value_new.txt') as f:
    data = f.read()
    value_new = json.loads(data)

with open('value_old.txt') as f:
    data = f.read()
    value_old = json.loads(data)

shared_items = {k: value_new[k] for k in value_new if k in value_old and value_new[k] == value_old[k]}
print(len(value_new))
print(len(shared_items))