# Import necessary packages
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import random
import os
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import Adam


# Source: •	https://keon.io/deep-q-learning/ (with significant adaptations)
# Deep Double Q-Learning Agent
class DDQNAgent:
    # Construct a new DDQNAgent
    def __init__(self, env):
        # Store the Acrobot-v1 environment
        self.env = env

        # Declare hyper-parameters
        # Declare gamma value
        self.gamma = 0.95
        # Declare initial, decay rate, and minimum epsilon value
        self.epsilon = 0.95
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.75
        # Declare learning rate for the multi-layer function approximator neural networks
        self.learning_rate = 0.001

        # Declare and create the multi-layer Q1 function approximator
        self.model_1 = self.create_model(env.action_space.n, self.learning_rate)
        # Declare and create the multi-layer Q2 function approximator
        self.model_2 = self.create_model(env.action_space.n, self.learning_rate)

    # Source: https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
    # Construct the Deep Double Q-Learning function approximation neural networks
    def create_model(self, num_actions, learning_rate):
        # Construct the multi-layer feed-forward function approximator neural network
        model = Sequential()
        # Declare dense layers
        model.add(Dense(48, input_shape=(240, 256, 3), activation="relu"))
        model.add(Dense(24, activation="relu"))
        # Flatten and output for Q(S, all 3 A)
        model.add(Flatten())
        model.add(Dense(7))
        # Compile model using mean squared error with the Adam optimizer
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        # Output the network summary
        # model.summary()

        # Return the created model
        return model

    # e-greedy algorithm to determine which action to take based on the DDQN network
    def determine_action(self, state, env):
        # Take a random action if the random number if less than or equal to epsilon
        if np.random.rand() <= self.epsilon:
            # Return a random legal action
            return env.action_space.sample()
        else:
            # Determine the maximal action from Q1 + Q2
            # image = tf.cast(state, tf.float32)
            image = tf.cast(np.reshape(state, (1, 240, 256, 3)), tf.float32)
            act_values_1 = self.model_1.predict(image)
            act_values_2 = self.model_2.predict(image)

            # Store Q1 + Q2 in separate list
            act_values_combined = act_values_1 + act_values_2

            # Return the highest predicted reward action in Q1 + Q2
            return np.argmax(act_values_combined)

    # Train the neural network using experience replay
    def replay(self, batch_size, memory, cp_1_callback, cp_2_callback):
        # Create a minibatch of previously observed samples
        if len(memory) < batch_size:
            return

        # Store minibatch samples
        minibatch = random.sample(memory, batch_size)

        # Loop for each sample in the minibatch
        for state, action, reward, next_state, done in minibatch:
            # Cast and reshape states
            state_cast = tf.cast(np.reshape(state, (1, 240, 256, 3)), tf.float32)
            next_state_cast = tf.cast(np.reshape(next_state, (1, 240, 256, 3)), tf.float32)

            # With probability .5 update Q1 and with probability .5 update Q2
            if np.random.rand() <= .5:
                # Update Q1
                # qsa1 = Q1(S, A)
                qsa1 = self.model_1.predict(state_cast)
                # qsaprime1 = Q1(S', a)
                qsaprime1 = self.model_1.predict(next_state_cast)
                # qsaprime1_maxa = argmax a of Q1(S', a)
                qsaprime1_maxa = np.argmax(qsaprime1[0])

                # qsaprime2 = Q2(S', a)
                qsaprime2 = self.model_2.predict(next_state_cast)

                # Q1(S, A) <- Q1(S, A) + (R + y*Q2(S', argmax a of Q1(S', a)) - Q1(S, A))
                target = qsa1[0][action] + (reward + self.gamma * qsaprime2[0][qsaprime1_maxa] - qsa1[0][action])
                qsa_target = qsa1
                qsa_target[0][action] = target

                # Train the updated model to fit the new qsa_target value (update network weights)
                self.model_1.fit(state_cast, qsa_target, epochs=1, verbose=0, callbacks=[cp_1_callback])
            else:
                # Update Q2
                # qsa2 = Q2(S, A)
                qsa2 = self.model_2.predict(state_cast)
                # qsaprime2 = Q2(S', a)
                qsaprime2 = self.model_2.predict(next_state_cast)
                # qsaprime2_maxa = argmax a of Q2(S', a)
                qsaprime2_maxa = np.argmax(qsaprime2[0])

                # qsaprime1 = Q1(S', a)
                qsaprime1 = self.model_1.predict(next_state_cast)

                # Q2(S, A) <- Q2(S, A) + (R + y*Q1(S', argmax a of Q2(S', a)) - Q2(S, A))
                target = qsa2[0][action] + (reward + self.gamma * qsaprime1[0][qsaprime2_maxa] - qsa2[0][action])
                qsa_target = qsa2
                qsa_target[0][action] = target

                # Train the updated model to fit the new qsa_target value (update network weights)
                self.model_2.fit(state_cast, qsa_target, epochs=1, verbose=0, callbacks=[cp_2_callback])

    # Decay the epsilon value by decay amount if it is not already at the minimum
    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Trains the agent using the Double Deep Q-Learning network
if __name__ == "__main__":
    # Configure gym environment and action space
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # Declare the agent
    agent = DDQNAgent(env)

    # Create memory storage for experience replay
    memory = deque(maxlen=2000)

    # Configure the random seeds for reproducability
    np.random.seed(100)
    env.seed(100)

    # Declare the number of episodes, steps per episode, and experience replay mini-batch size
    minibatch_size = 1
    score = 0
    done = True

    # Configure model saving
    checkpoint_1_path = "supermariobros_training/cp1.ckpt"
    checkpoint_1_dir = os.path.dirname(checkpoint_1_path)

    checkpoint_2_path = "supermariobros_training/cp2.ckpt"
    checkpoint_2_dir = os.path.dirname(checkpoint_2_path)

    cp_1_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_1_path,
                                                       save_weights_only=True,
                                                       verbose=1)

    cp_2_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_2_path,
                                                       save_weights_only=True,
                                                       verbose=1)

    for step in range(10000):
        if done:
            state = env.reset()
            score = 0
        env.render()

        # Determine action to take (Choose A from S using the policy e-greedy in Q1 + Q2)
        action = agent.determine_action(state, env)

        # Advance game to next state given action (Take action A, observe R, S')
        next_state, reward, done, info = env.step(action)

        # If the pendulum crosses the black bar, give a reward of 1000 to incentivize future behaviors
        if done:
            replay_reward = 1000
        else:
            replay_reward = reward

        # Remember the current information for experience replay
        memory.append((state, action, replay_reward, next_state, done))

        # Source: •	https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
        # Update Q1(S, A) or Q2(S, A) using Experience Replay
        agent.replay(minibatch_size, memory, cp_1_callback, cp_2_callback)

        # Update state (S <- S')
        state = next_state

        # Update score
        score += reward

        # Print score
        print("Step: {}, score: {}".format(step, score))

        # Decay the epsilon value after each episode
        agent.decay_epsilon()

    env.close()

    model.save('super_mario_bros_1.h5')
