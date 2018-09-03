import json
import sys
path_data = '../data/'
path_dest = '../data_perso/'


with open(path_dest+'data_Dev.json', 'r') as input:
    data_ref = json.load(input)
    input.close()

with open(path_dest+'data_toTest.json', 'r') as input:
    data_to_evaluate = json.load(input)
    input.close()

sum = 0
for id in data_ref:
    for position in data_ref[id]:
        if position == data_to_evaluate[id][0] :
            sum += 1
            break

print(sum / float(len(data_ref)))