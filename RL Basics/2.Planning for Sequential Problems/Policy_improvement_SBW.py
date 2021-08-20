from Slippery_Bandit_Walk import MDP_SBW
from Policy_improvement import policy_improvement
from Policy_evaluation import policy_evaluation
import numpy as np

# state:1 , action 1
#print(MDP_SBW[5][0])

pi = lambda s: {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0}[s]

# state:1 , action 1
#print(MDP_FL[5][0])

directions=dict()

directions[0]="left"
directions[1]="right"

# left:0 , right:1

V = policy_evaluation(pi,MDP_SBW)

improved_pi = policy_improvement(V, MDP_SBW)

for i in range(len(MDP_SBW)):
    print("action in state "+str(i)+" was "+directions[pi(i)]+ " and now is " +directions[improved_pi(i)])
V = policy_evaluation(improved_pi,MDP_SBW)


