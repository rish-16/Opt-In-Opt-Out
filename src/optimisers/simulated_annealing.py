import math
import numpy as np

class SimulatedAnnealingSolver:
    def __init__(self, Tmax, Tmin, Tdelta):
        super().__init__()
        self.Tmax = Tmax
        self.Tmin = Tmin
        self.Tdelta = Tdelta

    def run(self, init, f, mut, scrit=None):
        dist = np.arange(self.Tmax, self.Tmin, -self.Tdelta)
        prev = -math.inf
        allprevs = []

        for t in range(len(dist)):
            T = dist[t]

            if scrit:
                if scrit(init):
                    return init

            new_w = mut(init)
            score_w = f(new_w)
            
            mut_w = mut(new_w)
            score_mut = f(mut_w)

            if (score_mut > score_w):
                if (score_mut > prev):
                    init = mut_w
                    prev = score_mut
            else:
                if (np.exp(-(score_w) / T) > np.random.uniform(0, 1)):
                    if (score_w > prev):
                        init = new_w
                        prev = score_w
                else:
                    init = mut_w
                    prev = score_mut

        return init