import numpy as np
import numpy.random as npr

class DES:
    def __init__(self):
        self.agenda = {}
        self.time = 0
    def schedule(self, time, event, arguments):
        if time in self.agenda:
            self.agenda[time].append([event,arguments])
        else:
            self.agenda[time]=[[event,arguments]]
    def next(self):
        if len(self.agenda)>0:
            self.time = min(self.agenda)
            for event, args in self.agenda.pop(self.time):
                event(*args)
    def run(self, max_time = 100):
        steps = 0
        while len(self.agenda)>0:
            self.next()
            steps+=1
            if (self.time >= max_time): self.stop()
            
            
    def stop(self):
        self.agenda = {}

class Hawkes(DES):
    def __init__(self, lambda0, c, gamma, w):
        super().__init__()
        self.active_points = []
        self.lambda0 = lambda0
        self.c = c
        self.gamma = gamma
        self.w = w
        
        self.schedule(0, self.event, [])

        self.rejections = 0

    def phi(self, t):
        return self.c*np.exp(-self.gamma*t)*(0<=t)*(t<self.w)
    
    def event(self):
        self.active_points.append(self.time)
        self.active_points = [ti for ti in self.active_points if (self.time-ti)<self.w]

        t = self.time
        accepted = False

        upperbound=self.lambda0+len(self.active_points)*self.c

        while not accepted:
            t+=npr.exponential(1/upperbound)
            true_rate = self.lambda0 + sum( self.phi(t-ti) for ti in self.active_points)

            if npr.random()<true_rate/upperbound:
                accepted = True
            else:
                # Set upperbound for next iteration to last true rate as this rate is decreasing
                self.rejections += 1
                upperbound = true_rate

        self.schedule(t, self.event, [])
    
    def stop(self):
        print("REJECTIONS PER TIME UNIT {}".format(self.rejections/self.time))
        super().stop()
        

h = Hawkes(1, 0.5, 1, 1)
h.run(max_time=10000)