import statistics
import json
import math
import sys
import numpy as np

def print_statics(dataset):
    data_num_info = []
    data = {}
    with open(dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'data' in data:
        for pack in data['data']:
            data_num_info.append(len(pack))
    print(dataset)
    num_recp = len(data_num_info)
    print('  num of packs: {}'.format(num_recp))
    num_data = sum(data_num_info)
    print('  num of talks: {}'.format(num_data))

def label_statistics(dataset):
    sum_label = np.zeros(8)
    argmax_sum_label = np.zeros(8)
    data = {}
    with open(dataset, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if 'data' in data:
        for pack in data['data']:
            for talk in pack:
                label = talk['label']
                sum_label += np.array(label)
                argmax_sum_label[label.index(max(label))] += 1
    print('label sum:')
    print(sum_label)
    print(sum_label.sum())
    print('argmax label sum:')
    print(argmax_sum_label)
    print(argmax_sum_label.sum())

if __name__ == "__main__":
    print_statics(sys.argv[1])
    label_statistics(sys.argv[1])