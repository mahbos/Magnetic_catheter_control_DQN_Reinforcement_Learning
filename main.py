!pip install stable-baselines3
!pip install shimmy>=0.2.1
import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

class PlantControlEnv(gym.Env):
    def __init__(self, plant_model, input_scaler, output_scaler, desired_output):
        super(PlantControlEnv, self).__init__()

        self.plant = plant_model
        self.input_scaler = input_scaler
        self.output_scaler = output_scaler
        self.current_input = np.zeros((1,5))
        self.desired_output = desired_output

        self.action_space = spaces.Discrete(243)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

    def action_to_input_changes(self, action):
        changes = np.zeros(5)
        for i in range(5):
            changes[i] = (action % 3) - 1
            action //= 3
        return 0.1 * changes

    def step(self, action):
        input_changes = self.action_to_input_changes(action)
        self.current_input += input_changes

        # Clipping each input to its respective range
        self.current_input[0, 0] = np.clip(self.current_input[0, 0], -10, 10)
        self.current_input[0, 1] = np.clip(self.current_input[0, 1], 0, 60)
        self.current_input[0, 2] = np.clip(self.current_input[0, 2], 0, 70)
        self.current_input[0, 3] = np.clip(self.current_input[0, 3], -20, 90)
        self.current_input[0, 4] = np.clip(self.current_input[0, 4], -130, 130)

        scaled_input = self.input_scaler.transform(self.current_input)
        raw_output = self.plant.predict(scaled_input)
        output = self.output_scaler.inverse_transform(raw_output)[0]


    # ... [same as before]
        error = np.linalg.norm(output - self.desired_output)
        reward = -np.power(error, 2)
        # Terminate if error is small enough (for faster training)
        done = error < 0.05
        return output, reward, done, {}

    def reset(self):
        self.current_input = np.zeros((1,5))
        scaled_input = self.input_scaler.transform(self.current_input)
        raw_output = self.plant.predict(scaled_input)
        output = self.output_scaler.inverse_transform(raw_output)[0]
        return output

    def render(self, mode='human'):
        pass

    def close(self):
        pass

desired_output = np.array([0.5, -0.2, 0.8])  # Desired x, y, z values

env = DummyVecEnv([lambda: PlantControlEnv(plant, input_scaler, output_scaler, desired_output)])

# Creating the DQN model with logging and some parameter adjustments
model = DQN("MlpPolicy", env, learning_rate=5e-2, verbose=1, exploration_fraction=0.3, exploration_final_eps=0.1, tensorboard_log="./dqn_log")

#model = DQN("MlpPolicy", env, verbose=1, exploration_fraction=0.5, exploration_final_eps=0.05, tensorboard_log="./dqn_log")
model.learn(total_timesteps=5000)  # Increase the timesteps for more training

obs = env.reset()
errors = []
for i in range(100):
    action, _ = model.predict(obs)
    obs, _, _, _ = env.step(action)
    error = np.linalg.norm(obs - desired_output)
    errors.append(error)

# Plot the errors
plt.plot(errors)
plt.xlabel('Steps')
plt.ylabel('Error')
plt.show()
