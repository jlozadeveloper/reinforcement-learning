import numpy as np

class Evaluator:
    def __init__(self, agent, num_episodes: int):
        self.agent = agent
        self.num_episodes = num_episodes

    def evaluate(self):
        # self.agent.load("cartpole-dqn.keras")
        for episode in range(self.num_episodes):
            state, _ = self.agent.env.reset()
            state = np.reshape(state, [1, *self.agent.input_shape])
            done = False
            i = 0
            while not done:
                # self.agent.env.render()
                action = np.argmax(self.agent.model.predict(state, verbose=0))
                next_state, reward, terminated, truncated, _ = self.agent.env.step(action)
                done = terminated or truncated
                state = np.reshape(next_state, [1, *self.agent.input_shape])
                i += 1
                if done:
                    print("episode: {}/{}, score: {}".format(episode, self.num_episodes, i))
                    break