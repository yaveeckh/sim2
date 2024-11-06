import numpy as np
import numpy.random as npr
def hawkes_via_branching(lambda0,c,gamma,w,T):
        def phi(t):
            return c*np.exp(-gamma*t)*(0<=t)*(t<w)

        points=[]
        # generate points of generation 0
        current_gen = []
        
        t0 = npr.exponential(1/lambda0)
        current_gen.append(t0)
        
        while t0 < T:
            t0 += npr.exponential(1/lambda0)
            current_gen.append(t0)

        while len(current_gen) > 0:
            # merge current_gen into the set of points
            points.extend(current_gen)
            points.sort()

            # generate next_gen
            next_gen = []

            for tp in current_gen:
                tc = tp
                while tc < T and (tc-tp)<w:
                    tc += npr.exponential(1/c)
                    true_rate = phi(tc-tp)
                    if npr.random()<true_rate/c:
                        next_gen.append(tc)

            current_gen = next_gen

        return points


if __name__ == "__main__":
    h = hawkes_via_branching(1, 0.5, 1, 1, 100)
    print(h)