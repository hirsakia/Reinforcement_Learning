from Policy_evaluation import policy_evaluation
from Policy_improvement import policy_improvement
from Frozen_Lake import MDP_FL
import numpy as np


# state:1 , action 1
#print(MDP_FL[5][0])


directions=dict()

directions[0]="left"
directions[1]="right"
directions[2]="up"
directions[3]="down"



# left:0 , right:1, up:2, down:3

pi = lambda s: {0:1, 1:1, 2:3, 3:0, 4:3, 5:2, 6:3, 7:0, 8:1, 9:1, 10:3, 11:0, 12:0, 13:1, 14:1, 15:0}[s]


V = policy_evaluation(pi,MDP_FL)



improved_pi = policy_improvement(V, MDP_FL)



for i in range(len(MDP_FL)):
    print("action in state "+str(i)+" was "+directions[pi(i)]+ " and now is " +directions[improved_pi(i)])
V = policy_evaluation(improved_pi,MDP_FL)
