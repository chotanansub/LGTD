import json
import numpy as np

def generate_syn1(fn):
    np.random.seed(0)
    x = np.arange(0.1,2*np.pi,0.025)
    T = len(x) * 2
    one_period = np.zeros(len(x)*2)
    one_period[:len(x)] = - np.sin(x)
    one_period[len(x):] = 0
    seasonal = []
    for i in range(20):
        seasonal = np.concatenate([seasonal, one_period])
        trend = np.zeros(720*4)
        trend = np.concatenate([trend, np.ones(270*4)])
        trend = np.concatenate([trend, 2 * np.ones(430*4)])
        trend = np.concatenate([trend, 3 * np.ones(550*4)])
        trend = np.concatenate([trend, 2 * np.ones(510*4)])
    residual = 0.03*np.random.randn(len(trend))
    syn1 = seasonal + trend + residual
    data = {}
    data['period'] = T
    data['trainTestSplit'] = 5 * T
    data['ts'] = list(syn1)
    data['trend'] = list(trend)
    data['seasonal'] = list(seasonal)
    data['residual'] = list(residual)
    with open(fn, "w") as outfile:
        json.dump(data, outfile)

def generate_syn2(fn, n=20):
    np.random.seed(1)
    x = np.concatenate([5*np.ones(100), -5*np.ones(100)])
    T = len(x)
    one_period = x
    shift_period = np.zeros(one_period.shape)
    shift = 10
    shift_period[:shift] = -5
    shift_period[shift:] = one_period[:-shift]
    seasonal = []
    for i in range(n):
        if i <= 4:
            seasonal = np.concatenate([seasonal, one_period])
        else:
            if np.random.rand() > 0.2:
                seasonal = np.concatenate([seasonal, one_period])
            else:
                seasonal = np.concatenate([seasonal, shift_period])
    trend = np.zeros(len(seasonal))
    residual = 0.03*np.random.randn(len(trend))
    syn2 = seasonal + trend + residual
    data = {}
    data['period'] = T
    data['trainTestSplit'] = 5 * T
    data['ts'] = list(syn2)
    data['trend'] = list(trend)
    data['seasonal'] = list(seasonal)
    data['residual'] = list(residual)
    with open(fn, "w") as outfile:
        json.dump(data, outfile)

generate_syn1('syn1.json')
generate_syn2('syn2.json')

import json

with open('/content/syn1.json', 'r') as f:
    syn1_data = json.load(f)

print(syn1_data.keys())
#output: dict_keys(['period', 'trainTestSplit', 'ts', 'trend', 'seasonal', 'residual'])

len(syn1_data['trend'])
#output: 9920

import os
cmd = 'java -jar /content/OneShotSTL/java/OneShotSTL/OneShotSTL.jar --method OneShotSTL --task decompose --shiftWindow 0 --in /content/syn1.json --out /content/result/syn1_OneShotSTL.json'
_ = os.system(cmd)
cmd = 'java -jar /content/OneShotSTL/java/OneShotSTL/OneShotSTL.jar --method OneShotSTL --task decompose --shiftWindow 10 --in /content/syn2.json --out /content/result/syn2_OneShotSTL.json'
_ = os.system(cmd)

with open('/content/result/syn1_OneShotSTL.json', 'r') as f:
    syn1_result = json.load(f)

syn1_result.keys()
#output: dict_keys(['trend', 'seasonal', 'residual'])

len(syn1_result['trend'])
#output: 7440