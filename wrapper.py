
class envWrapper(object):
    def __init__(self, agents_property):

        self.agents_property = agents_property
        self.agent_dict = {}
        # self.envs = {}

        self.steps = 0

        self.viewer = None
        self.createAgents()


    @property
    def agents(self):
        return [agent for grouplist in self.agent_dict.values() for agent in grouplist]

    def reset(self, group):
        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """
        state = []
        for agent in self.agent_dict[group]:
            state.append(agent.reset())
        # state = self.envs[group].reset()
        return state

    def createAgents(self):
        for group in self.agents_property:
            al = []
            for i in range(self.agents_property[group]['init_n']):
                if type(self.agents_property[group]['env']) == str:
                    import gym
                    a = gym.make(self.agents_property[group]['env'])
                else:
                    a = self.agents_property[group]['env']()
                al.append(a)
            self.agent_dict[group] = al
        # for group in self.agents_property:
        #     self.envs[group] = SubprocVecEnv(self,al)

    def step(self, group, actions):
        state = []
        assert len(self.agent_dict[group]) == len(actions)
        for agent, action in zip(self.agent_dict[group],actions):
            state.append(agent.step(action))
        # s, r, done, info = self.envs[group].step(actions)
        self.steps += 1
        return zip(*state)

    def render(self, mode='human'):
        """Renders the environment.
        Args:
            mode (str): the mode to render with
            close (bool): close all open renderings
        """
        # render only first agents
        for agents in self.agent_dict:
            self.agent_dict[agents][0].render()
            break
