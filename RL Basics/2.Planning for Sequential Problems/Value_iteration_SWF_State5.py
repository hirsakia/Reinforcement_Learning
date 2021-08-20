from Policy_improvement import MDP_SBW
import numpy as np

v=np.zeros((101,7))
print(v)

# always go left

p_left=0.5
p_stay=0.333
p_right=0.166
epsilon=1e-6
for i in range(100):

    v[i+1][5]=p_left*(1+v[i][4])+p_stay*(0+v[i][5])+p_right*(0+v[i][6])
    v[i+1][4]=p_left*(0+v[i][3])+p_stay*(0+v[i][4])+p_right*(0+v[i][5])
    v[i+1][3]=p_left*(0+v[i][2])+p_stay*(0+v[i][3])+p_right*(0+v[i][4])
    v[i+1][2]=p_left*(0+v[i][1])+p_stay*(0+v[i][2])+p_right*(0+v[i][3])
    v[i+1][1]=p_left*(0+v[i][0])+p_stay*(0+v[i][1])+p_right*(0+v[i][2])
    print(v[i+1][5])
    if (v[i+1][5]-v[i][5]) < epsilon:
        print(i)
        break




