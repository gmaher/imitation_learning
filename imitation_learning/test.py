class TestEnv:
    def __init__(self, right=4):
        self.start = int(right/2)
        self.state = self.start
        self.left  = 0
        self.right = right
        self.n     = right+1

    def step(self, action):
        r = 0
        done = False
        if action == 0:
            self.state -= 1
        if action == 1:
            self.state += 1

        if self.state == self.left:
            r = 0
            done = True

        if self.state == self.right:
            r = 1
            done = True

        return self.state, r, done, {}

    def reset(self):
        self.state = self.start
        return self.state
