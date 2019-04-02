import sys
from copy import deepcopy
import numpy as np

grid = np.loadtxt(sys.argv[1], dtype=int)
nr = len(grid)
nc = len(grid[0])
f=open(sys.argv[2], 'r')
p = float(sys.argv[3])
x = f.readline().split()
policy=[]
value=[]
while(x[0]!="iterations"):
	policy.append(int(x[1]))
	value.append(float(x[0]))
	x = f.readline().split()

iter = x[1]

s=0
a=4
end=[]
state_to_pos = {}
pos_to_state = {}
st=0
for i in range(nr):
	for j in range(nc):
		if grid[i][j]==0:
			state_to_pos[s]=(i,j)
			pos_to_state[(i,j)]=s
			s+=1
		elif grid[i][j]==2:
			state_to_pos[s]=(i,j)
			pos_to_state[(i,j)]=s
			st=s
			s+=1
		elif grid[i][j]==3:
			state_to_pos[s]=(i,j)
			pos_to_state[(i,j)]=s
			end.append(s)
			s+=1

state=st
direc=['N', 'E', 'W', 'S']
dx=[-1,0,0,1]
dy=[0,1,-1,0]
while True:
	a=policy[state]
	actions=[a]
	dist = [p]
	(x,y) = state_to_pos[state]
	for i in range(4):
		if(0<= x+dx[i]<nr and 0<=y+dy[i]<nc and grid[x+dx[i]][y+dy[i]]!=1):
			actions.append(i)
	dist = dist+[(1.0-p)/(len(actions)-1)]*(len(actions)-1)
	a = np.random.choice(actions, p=dist)
	print(direc[a], end='')
	(x,y) = state_to_pos[state]
	state = pos_to_state[(x+dx[a], y+dy[a])]
	if state in end:
		break
	else:
		print("", end=' ')
print()