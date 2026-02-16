class FixedTimeController:
    def __init__(self, cycle: int = 20):
        self.cycle = int(cycle)
        self.t = 0
        self.phase = 0

    def reset(self):
        self.t = 0
        self.phase = 0

    def act(self, obs):
        self.t += 1
        if self.t % self.cycle == 0:
            self.phase = 1 - self.phase
        return int(self.phase)

class FixedTimeGridController:
    def __init__(self, cycle: int = 20):
        self.cycle = int(cycle)
        self.t = 0
        self.phase = 0

    def reset(self):
        self.t = 0
        self.phase = 0

    def act(self, obs):
        self.t += 1
        if self.t % self.cycle == 0:
            self.phase = 1 - self.phase
        a = 0
        for i in range(4):
            a |= (self.phase << i)
        return int(a)
