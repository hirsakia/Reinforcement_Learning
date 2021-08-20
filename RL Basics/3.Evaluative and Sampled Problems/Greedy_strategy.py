from epsilon_greedy import epsilon_greedy
from Pure_exploitation import pure_exploitation
from Pure_exploration import pure_exploration

import numpy as np

MDP_SBW=dict()

# state=action(probability,state,reward,terminal_or_nonterminal)
# left:0 , right:1

MDP_SBW[0]={0: [(1.0, 0, 0.0, True)], 1:[(1.0, 0, 0.0, True)]}
MDP_SBW[1]={0: [(0.8, 0, -1.0, True),  (0.2, 0, 100.0, True)],\
     1: [(0.8, 2, 100.0, True), (0.2, 0, -1.0, True)]}
MDP_SBW[2]={0: [(1.0, 2, 0.0, True)], 1:[(1.0, 2, 0.0, True)]}
'''
for i in range(5):
    name, returns, Qe, actions,action_one, action_zero=pure_exploitation(MDP_SBW,10000)
    print(returns)
    print(actions)
    print(Qe)
    print("action one "+str(action_one)) 
    print("action zero "+str(action_zero)) 
'''
'''
for i in range(5):
    name, returns, Qe, actions,action_one, action_zero=pure_exploration(MDP_SBW,10000)
    print(returns)
    print(actions)
    print(Qe)
    print("action one "+str(action_one)) 
    print("action zero "+str(action_zero)) 
'''
for i in range(5):
    name, returns, Qe, actions,action_one, action_zero=epsilon_greedy(MDP_SBW,0.001,1000)
    print(np.sum(returns))
    print(actions)
    print(Qe)
    print("action one "+str(action_one)) 
    print("action zero "+str(action_zero)) 