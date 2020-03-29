import numpy as np
from scipy.optimize import linprog


class LP:
	def __init__(self, a, b, c):
		self.a = a
		self.b = b
		self.c = c

def vg_method(c, a, b):
	ans = None
	queue = [LP(a, b, c)]
	while queue:
		s = queue.pop(0)
		res = linprog(s.c, A_ub=s.a, b_ub=s.b, method="simplex")
		print('Optimal value:', res.fun, '\nX:', res.x)
		if ans != None and ans.fun <= res.fun:
			continue
		found = False
		for i in range(len(s.a[0])):
			if int(res.x[i]) != res.x[i]:
				found = True
				tmp = [0] * len(s.a[0])
				tmp[i] = 1
				queue.append(LP(
					s.a + [tmp],
					s.b + [int(res.x[i])],
					s.c)
				)

				tmp2 = [0] * len(s.a[0])
				tmp2[i] = -1
				queue.append(LP(
					s.a + [tmp2],
					s.b + [-(int(res.x[i]) + 1)],
					s.c)
				)

		if not found and (ans == None or ans.fun > res.fun):
			ans = res
	return ans






