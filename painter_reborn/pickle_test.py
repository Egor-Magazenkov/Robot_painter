import numpy as np
import pandas as pd
import os
import traceback
import time
import re
import pickle
import sys
import json
import matplotlib.pyplot as plt 

directory = './pickles/'
filename = directory + 'trjs_bridge.pickle'

with open(filename, 'rb') as file:
    data = pickle.load(file)

    trajectories = data['trajectories']

   

    for i, trajectory in enumerate(trajectories[1:2]):
        points = trajectory["points"]
        print(trajectory)
        plt.scatter(points[:, 0], points[:, 1])

plt.show()