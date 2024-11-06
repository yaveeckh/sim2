import numpy.random as npr
import numpy as np

def hawkes(lambda0, c, gamma, w, t0=0):
    def phi(t):
        return c*np.exp(-gamma*t)*(0<=t)*(t<w)
    t=t0
    active_points=[]
    while True:
        upperbound=lambda0+len(active_points)*c
        t+=npr.exponential(1/upperbound)
        true_rate = lambda0 + sum( phi(t-ti) for ti in active_points)
        if npr.random()<true_rate/upperbound:
            yield t
            active_points.append(t)
            # prune active points when outside window
            active_points = [ti for ti in active_points if (t-ti)<w]
        

h = hawkes(1, 1, 1, w=1)

for i in range(10):
    print(h.__next__())
