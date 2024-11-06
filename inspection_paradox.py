import hawkes_branching as hb
import numpy as np
import numpy.random as npr

parameters = {
    "lambda0": 1,
    "c": 0.5,
    "gamma": 1,
    "w": 1,
    "T": 100000
}

def iet_estimator():
    points = hb.hawkes_via_branching(**parameters)
    iet = np.diff(points)
    return np.mean(iet)

def mtunp_estimator1(L):
    
    points = hb.hawkes_via_branching(**parameters)
    M = len(points)//L
    batch_means = []
    for i in range(M):
        batch_points = points[i*L:min(len(points), (i+1)*L)]
        # Poisson observer points
        observer_points = []
        t = points[max(0,i*L-1)] + npr.exponential(1/parameters["lambda0"])

        while t < batch_points[-1]:
            observer_points.append(t)
            t += npr.exponential(1/parameters["lambda0"])

        tunp = []

        for op in observer_points:
            sup = min([i for i in batch_points if i>op])
            
            tunp.append(sup - op)
        
        batch_means.append(np.mean(tunp))
    return np.mean(batch_means)

    
def mtunp_estimator2():
    
    points = hb.hawkes_via_branching(**parameters)
    iet = np.diff(points)
    
    miet = np.mean(iet)
    msiet = np.mean(iet**2)

    return msiet/(2*miet)

if __name__ == "__main__":
    print(iet_estimator())
    print(mtunp_estimator1(100))
    print(mtunp_estimator2())