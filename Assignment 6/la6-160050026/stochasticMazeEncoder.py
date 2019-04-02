import sys
from copy import deepcopy
import numpy as np

reward1=-1.0
reward2=100.0
gamma=0.9

p = float(sys.argv[2])
def reward(x):
	if(x==3):
		return reward2
	else:
		return reward1

grid = np.loadtxt(sys.argv[1], dtype=int)
nr = len(grid)
nc = len(grid[0])
s=0
a=4
end=[]
states = {}
st=0
for i in range(nr):
	for j in range(nc):
		if grid[i][j]==0:
			states[(i,j)]=s
			s+=1
		elif grid[i][j]==2:
			states[(i,j)]=s
			st=s
			s+=1
		elif grid[i][j]==3:
			states[(i,j)]=s
			end.append(s)
			s+=1
print("numStates", s)
print("numActions", a)
print("start", st)
print("end", end=' ')
for i in end:
	print(i, end=' ')
print()

dx=[-1,0,0,1]
dy=[0,1,-1,0]
for (x,y), state in states.items():
	actions=[]
	for i in range(4):
		if(0<= x+dx[i]<nr and 0<=y+dy[i]<nc and grid[x+dx[i]][y+dy[i]]!=1):
			actions.append(i)
	for action in actions:
		# print(states[(x+dx[i],y+dy[i])])
		print("transition", state, action, states[(x+dx[action],y+dy[action])], reward(grid[x+dx[action]][y+dy[action]]), p)
		for action1 in actions:
			print("transition", state, action, states[(x+dx[action1],y+dy[action1])], reward(grid[x+dx[action1]][y+dy[action1]]), (1.0-p)/len(actions))

print("discount", gamma)