#!/usr/bin/python3

from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

with open('diabetes_data.txt', mode='w') as f:
    for dd in diabetes['data']:
        for d in dd:
            f.write(str(d))
            f.write(' ')
        f.write('\n')

with open('diabetes_target.txt', mode='w') as f:
    for d in diabetes['target']:
        f.write(str(d))
        f.write('\n')

