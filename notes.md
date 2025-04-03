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

- Right now our model is plateauing, it is not exploring new options and whenever and keeps the same reward, no crashes, and same steps
- There is a lack of exploration / exploitation balance because our epsilon decays too quickly to 0.01 taking the same actions 99% of the time
- It found one strategy that works and stuck with it, so it got stuck in a local maxima
- Our neurons are stuck
- We are using a small network so it might not capturing complex patterns
- Our standard DQN is overestimating our q-values which leads to an optimistic action selection and stay on the "safe" actions as a result
- There are only 3 possible discrete actions our model can take. It cannot turn or change its speed well
- Small replay buffer so it might not maintain alot of different experiences 

### Potential solutions

Below are some brainstormed solutions we could use based on the questions I raised before:
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

First of all, we could do some hyperparameter tuning. I mentioned before one of the problems was that there is a lack of exploration / exploitation balance. To improve it we can slow down epsilon decay from `0.995` to `0.997`. Then we can increase the minimum epsilon from `0.01` to `0.05`.

Then, to make our learning more stable we could use a Dobule DQN to prevent overestimation. Then we could change from using `MSE` to `Huber loss` which is more robust. Finally, we can reduce the learning rate to avoid missing better solutions and not always taking the "safe" strategy while training. We can reduce it from `0.001` to `0.0005`.

Another potential problem could be that the number of discrete actions in our model is not enough. If we give more actions to our model we could make it more efficient because it would increase the number of possibilities at each given state for our model. The first option will be to introduce a new `BRAKE` action where the car can stop while racing. We could also make the turning continuous instead of discrete with the car angle only increasing or decreasing by 5 but we can test it after. Hence, we will increaase the discrete action space from 3 -> 4.

Finally, to fix our small network problem we can increase our input layer from 24 -> 64, and our activation layer from 24 -> 32. Then, to prevent our dying neurons problem from `ReLU` we can use `LeakyReLU` to prevent this.

>| What would hapen if you change it to a continous action space with turning with +/- 1