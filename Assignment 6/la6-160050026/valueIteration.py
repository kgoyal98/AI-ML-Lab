import sys
import math

eps = 1e-16
x = input().split()
assert(x[0]=='numStates')
s = int(x[1])
x = input().split()
assert(x[0]=='numActions')
a = int(x[1])
x = input().split()
assert(x[0]=='start')
st = int(x[1])
x = input().split()
assert(x[0]=='end')
end = [int(t) for t in x[1:]]
x = input().split()
t = [{} for _ in range(a)]
while (x[0]=='transition'):
	if not int(x[1]) in t[int(x[2])].keys():
		t[int(x[2])][int(x[1])] = []
	t[int(x[2])][int(x[1])].append((int(x[3]), float(x[4]), float(x[5])))
	x = input().split()
assert(x[0]=='discount')
gamma = float(x[1])

value = [0.0]*s
policy = [0]*s
iter=0
while True:
	# print(iter+1)
	value1 = [-math.inf]*s
	for state in range(s):
		if state in end:
			value1[state]=0.0
			policy[state]=-1
			continue
		for action in range(a):
			# print(t)
			if(not state in t[action].keys()):
				continue
			v=0.0
			for (s1,r,p) in t[action][state]:
				v+=p*(r+gamma*value[s1])
			if(value1[state] < v):
				value1[state] = v
				policy[state] = action
	iter+=1
	terminate=True
	for i in range(s):
		if(abs(value[i]-value1[i])>eps):
			terminate=False
	value = value1
	if(terminate):
		break
for i in range(s):
	print(value[i], policy[i])
print("iterations", iter)



