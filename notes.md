- Remember in a DQN you use one
- 

### What we already have

First of all this is the model:
```python
def _build_model(self):
        # Neural Network for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=LEARNING_RATE))
        return model
```

We are using 3 layers in our DQN modal. Where we havea an input layer with 24 neurons and a ReLu activatin function

>| Maybe we could change the activation function here? How would it change our model performance?

Then we have one hidden layer and an output layer that has neurons equal to the number of action. It is like this because each neuron represents the possible q-value for one specific action, then based on expected total reward for each one our RL model which choose it.

The way we take actions is as follows:
```python
def act(self, state, explore_rate=0.0):
    # Epsilon-greedy action selection
    if np.random.rand() <= explore_rate:
        return random.randrange(self.action_size)
    
    state_tensor = np.array(state).reshape(1, self.state_size)
    act_values = self.model.predict(state_tensor, verbose=0)
    return np.argmax(act_values[0])
```

It uses epsilon greedy here to choose the action where it usees exploration using a random number between 0 or 1 based on if its the `explore_rate`. If not, then use the model to exploit our answer by predicting and returning the highest q value

>| How would changing the action function affect the model? What are some possible functions I could use?

Finally, we need to look at how we are using experience replay.

```python
def replay(self, batch_size):
    # Train the model with randomly sampled batch from memory
    if len(self.memory) < batch_size:
        return
    
    minibatch = random.sample(self.memory, batch_size)
    states = np.array([experience[0] for experience in minibatch])
    actions = np.array([experience[1] for experience in minibatch])
    rewards = np.array([experience[2] for experience in minibatch])
    next_states = np.array([experience[3] for experience in minibatch])
    dones = np.array([experience[4] for experience in minibatch])
    
    # Calculate Q values
    states_tensor = np.array(states)
    next_states_tensor = np.array(next_states)
    
    targets = self.model.predict(states_tensor, verbose=0)
    next_q_values = self.target_model.predict(next_states_tensor, verbose=0)
    
    for i in range(batch_size):
        if dones[i]:
            targets[i][actions[i]] = rewards[i]
        else:
            targets[i][actions[i]] = rewards[i] + GAMMA * np.amax(next_q_values[i])
    
    # Train the model
    self.model.fit(states_tensor, targets, epochs=1, verbose=0)
    self.training_count += 1
    
    # Decay epsilon
    if self.epsilon > EPSILON_MIN:
        self.epsilon *= EPSILON_DECAY
```

I takes a random batch of experiences from memory. It then uses the main model to get the current q values. Then, it gets the next state q-values from the target model. Then it looks at each experience in the batch. If it is done, it only assigns the target to a reward. If not, it assigns the target with the reward + (discount factor x max future Q-value). That is the Bellman equation

>| Maybe I could change the equation used to append rewards. How would changing the reward equation affect the results?

Finally, at the end we update the weights of the models and decrease exploration rate so that instead of learning from current experiences it learns from random past experiences. This helps the model break from temporal correlations and learn efficiently.

### The problem our model keeps encountering



### Potential solutions

- Different inputs

>| Maybe we could change the activation function here? How would it change our model performance?
- LeakyReLU
- Tanh
- Sigmoid
- ELU

>| How would changing the action function affect the model? What are some possible functions I could use?
- Have range of actions instead of distinct ones
- Introduce a "BRAKE" action or something else

>| Maybe I could change the equation used to append rewards. How would changing the reward equation affect the results?
- SARSA
- Double DQN
- Dueling DQN 
