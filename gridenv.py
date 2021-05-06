# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:27:57 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 20:01:22 2021

@author: Ramesh
"""

import numpy as np
import torch

class GridWorld():
    def __init__(self,row,column):
        self.i = row
        self.j = column
        self.gridwrld = np.zeros((self.i,self.j))
        self.block = [155, 205, 255, 305, 355, 405, 455, 220, 270, 320, 370, 
              420, 470, 520, 570, 620, 670, 720, 770, 820, 870,
              920, 970, 1020, 1070, 1120, 1170, 1220, 1270, 1320,
              1370, 1420, 1470, 1520, 1570, 1620, 1670, 1720, 1770,
              1820, 1870, 1920, 1970, 566, 616, 666, 716, 766, 816,
              866, 916, 966, 1016, 1066, 1116, 1166, 1216, 1266,
              1454, 1455, 1456, 1457, 1458, 1459, 1460, 1714, 1764,
              1814, 1864, 1914, 1964, 2014, 2064, 2114, 2164,
              801, 802,	803, 804, 805, 806, 807, 808, 809, 810, 811,
              812, 813, 1424, 1425, 1426, 1427, 1428, 1429, 1430, 
              1431, 1432, 1433, 1434, 229, 279, 329, 379, 429, 479, 
              529, 579, 629, 679, 729, 779, 829, 879, 929, 979,
              1029, 1079, 1129, 1179, 1229, 1787, 1837, 1887, 1937,
              1987, 1940, 1990, 2040, 2090, 2140, 2190, 2240, 2290,
              2337, 2338, 2390, 2391, 2399, 1034, 1035, 1036, 1037, 
              1038, 1039, 1040, 1041, 1042, 1043, 1044]
        self.statespace =  [states for states in range(self.i * self.j)]
        self.statespace.remove(self.i * self.j -1)
        self.statespaceadd = [states for states in range(self.i * self.j)]
        self.statespace_withWall = [states for states in self.statespaceadd if states not in self.block]
        
        self.actions = { 'U' : -self.i, 'D': self.i,
                         'L': -1, 'R': 1}
        self.end_state = (self.i * self.j -1) #Endstate of the grid
        self.terminal = False
        self.state_space = [state for state in range(self.i*self.j)]
        self.walls = [state for state in self.state_space if state not in self.block]
        self.positon_row= 0
        self.position_column = 0
        self.position = 0
        self.state = self.i*self.j
                    
    def off_grid_move (self, new_state, old_state):
        if new_state not in self.statespace_withWall:
            return True
        elif old_state % self.i == 0 and new_state % self.i == self.i -1:
            return True
        elif old_state % self.i == self.i -1 and new_state % self.i == 0:
            return True
        else:
            return False
        
    def one_hot_obs(self,sta):  
        sta =torch.tensor(sta)
        observation = torch.nn.functional.one_hot(sta, num_classes = self.state)
        return observation.float()
        
    def step(self,action):
        self.terminal = False
        self.positon_row = int(self.position/self.i)
        self.position_column = self.position%self.j
        next_state = self.position + self.actions[action]
        if next_state == self.end_state:
            self.terminal = True
            rew = 1
        else: 
            rew = -1
        if any(block ==next_state for block in self.block):
            return self.position, rew, self.terminal, None
        elif not self.off_grid_move(next_state, self.position):
            self.gridwrld[self.positon_row][self.position_column] = 0
            self.position = next_state
            self.positon_row = int(self.position/self.i)
            self.position_column = self.position%self.j
            self.gridwrld[self.positon_row][self.position_column] = 1
            return next_state, rew, self.terminal, None
        else:
            return self.position, rew, self.terminal, None
        
        
    def reset(self):
        self.position = 0
        obse = torch.zeros(1)
        self.gridwrld = np.zeros((self.i,self.j))
        return obse
        
        