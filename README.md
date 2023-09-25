Magnetic Catheter Control with 5-DOF Manipulator Using DQN
Description

This repository provides an implementation for controlling a magnetic catheter's coordinates (x, y, z) using a 5-degree-of-freedom (DOF) manipulator carrying an external permanent magnet. The control strategy employs a Deep Q-Network (DQN) agent trained on a custom environment built on the OpenAI Gym framework.

Features

Custom Environment (PlantControlEnv): Represents the dynamics of the 5-DOF manipulator and its interaction with the magnetic catheter.
Action Space: Designed to manipulate five parameters (corresponding to the 5 DOFs) of the manipulator.
Scalers: Use of input_scaler and output_scaler to standardize input actions and output coordinates.
Reinforcement Learning: Uses stable_baselines3 to implement and train the DQN agent for the control task.
Error Visualization: Plots the error between the desired and achieved catheter coordinates over time.
Requirements

gym
numpy
matplotlib
stable-baselines3
shimmy (version >= 0.2.1)
How to Use

Ensure that you have the required libraries installed using the pip install commands at the start of the code.
Define and instantiate the plant (representing the 5-DOF manipulator), input_scaler, and output_scaler.
Run the script. It initializes the environment, trains the DQN agent, and subsequently visualizes the control error.
