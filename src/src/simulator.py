import time
class Simulator(object):
    def __init__(self, env, agent, discount=1.0):
        self.env      = env
        self.agent    = agent
        self.discount = discount

    def run(self, render=False, num_steps=100):
        reward = 0
        self.tuples = []

        s = self.env.reset()

        for t in range(num_steps):
            if render:
                self.env.render()
                time.sleep(1.0/60)

            a = self.agent.act(s)

            ss,r,done,info = self.env.step(a)

            self.tuples.append((s,a,r,ss,done))

            reward += r*self.discount**t

            s = ss

            if done:
                break

        #print("{}: R={:.2f}, done={}".format(t,reward,done))

        return reward
