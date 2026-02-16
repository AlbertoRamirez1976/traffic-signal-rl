class MaxQueueController:
    def reset(self):
        pass

    def act(self, obs):
        q_ns, q_ew = float(obs[0]), float(obs[1])
        return 0 if q_ns >= q_ew else 1

class MaxQueueGridController:
    def reset(self):
        pass

    def act(self, obs):
        a = 0
        for i in range(4):
            q_ns = float(obs[i*4 + 0])
            q_ew = float(obs[i*4 + 1])
            phase = 0 if q_ns >= q_ew else 1
            a |= (phase << i)
        return int(a)
