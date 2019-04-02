import sys
from copy import deepcopy
import numpy as np

reward1=-1.0
reward2=100.0
gamma=0.9

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
	for i in range(4):
		if(0<= x+dx[i]<nr and 0<=y+dy[i]<nc):
			if(grid[x+dx[i]][y+dy[i]]==0):
				print("transition", state, i, states[(x+dx[i],y+dy[i])], reward1, 1)
			elif(grid[x+dx[i]][y+dy[i]]==1):
				print("transition", state, i, state, reward1, 1)
			elif(grid[x+dx[i]][y+dy[i]]==2):
				print("transition", state, i, states[(x+dx[i],y+dy[i])], reward1, 1)
			elif(grid[x+dx[i]][y+dy[i]]==3):
				print("transition", state, i, states[(x+dx[i],y+dy[i])], reward2, 1)
		else:
			print("transition", state, i, state, reward1, 1)

print("discount", gamma)