import gym
import retro
import random
import numpy as np

import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Flatten, Dense, Input, MaxPooling2D
from keras.models import Model, Sequential

from statistics import mean, median
from collections import Counter

" NOTES TO SELF! "
"""
Action space er kun 1 ad gangen - problemet er, at den operer med alle inputs per game og ikke per move?
"""

LR = 1e-3
env = retro.make(game='Breakout-Atari2600', state='Start')
#env = gym.make('CartPole-v0')
env.reset()
goal_steps = 200

score_requirement = 1
initial_games = 20

def some_random_games_first():
    for episode in range(5):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            print(action)
            observation, reward, done, info = env.step(action)
            if done:
                break

#some_random_games_first()

def initial_population():
    training_data = [] # observations made
    scores = [] #
    accepted_scores = []
    for game in range(initial_games):
        print('Playing game %s out of %s' % (game, initial_games))
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            #env.render()
            action = np.random.randint(0, 2, env.action_space.n, dtype=env.action_space.dtype)
            #action = np.append(random.randrange(0,2), np.repeat(0, 7))
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0: # if there was a previous observation we could have made
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward

            if done:
                break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                training_data.append([data[0], data[1]])

        env.reset()
        scores.append(score)
    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score: %s' % mean(accepted_scores))
    print('Median accepted score: %s' % median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

training_data = initial_population()
#def build_keras_model(training_data):
with tf.device("/cpu:0"): # replace with CPU if using CPU keras and TF

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=env.observation_space.shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(8, activation='softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    x = [i[0] for i in training_data] #np.array(i[0] for i in train_data).reshape(-1, len(train_data[0][0]), 1)
    y = [i[1] for i in training_data]

    model.fit(np.array(x), np.array(y), epochs=1, batch_size=32)
    #return model


#def play_model(model):
scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = np.random.randint(0, 2, env.action_space.n, dtype=env.action_space.dtype)
        else:
            action = model.predict(np.array(np.expand_dims(prev_obs, axis=0)))
        choices.append(action)

        new_observation, reward, done, info = env.step(action.ravel().astype('int8'))
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break

    scores.append(score)

print('Average Score %s' % (sum(scores)/len(scores)))
    #print('Choice 1: {},Choice')

#training_data = initial_population()
#model = build_keras_model(training_data)
#play_model(model)
# Play the game with fitted model


""" not impl. yet
def build_network(num_actions, agent_history_length, resized_width, resized_height, name_scope):
    with tf.device("/cpu:0"):
        with tf.name_scope(name_scope):
            state = tf.placeholder(tf.float32, [None, agent_history_length, resized_width, resized_height],
                                   name="state")
            inputs = Input(shape=(agent_history_length, resized_width, resized_height,))
            model = Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), activation='relu', padding='same',
                           data_format='channels_first')(inputs)
            model = Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu', padding='same',
                           data_format='channels_first')(model)
            # model = Conv2D(filter=64, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same')(model)
            model = Flatten()(model)
            model = Dense(256, activation='relu')(model)
            print model
            q_values = Dense(num_actions)(model)

            # UserWarning: Update your `Model` call to the Keras 2 API: 
            # `Model(outputs=Tensor("de..., inputs=Tensor("in..
            m = Model(inputs=inputs, outputs=q_values)

    return state, m
"""
