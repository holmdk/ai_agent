
# coding: utf-8

# In[5]:


import gym
import numpy as np
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils, np_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input


# In[6]:


def downsample(image):
    # Take only alternate pixels - basically halves the resolution of the image (which is fine for us)
    return image[::2, ::2, :]

def remove_color(image):
    """Convert all color (RGB is the third dimension in the image)"""
    return image[:, :, 0]

def remove_background(image):
    image[image == 144] = 0
    image[image == 109] = 0
    return image

def preprocess_observations(input_observation, prev_processed_observation, input_dimensions):
    """ convert the 210x160x3 uint8 frame into a 6400 float vector """
    processed_observation = input_observation[35:195] # crop
    processed_observation = downsample(processed_observation)
    processed_observation = remove_color(processed_observation)
    processed_observation = remove_background(processed_observation)
    processed_observation[processed_observation != 0] = 1 # everything else (paddles, ball) just set to 1
    #print(processed_observation.shape)
    # Convert from 80 x 80 matrix to 1600 x 1 matrix
    #processed_observation = processed_observation.astype(np.float).ravel()

    # subtract the previous frame from the current one so we are only processing on changes in the game
    if prev_processed_observation is not None:
        input_observation = processed_observation - prev_processed_observation
    else:
        input_observation = np.zeros(input_dimensions)
    # store the previous frame so we can subtract from it next time
    prev_processed_observations = processed_observation
    #print(prev_processed_observation)
    return input_observation, prev_processed_observations


# In[7]:


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector

def apply_neural_nets(observation_matrix, weights):
    """ Based on the observation_matrix and weights, compute the new hidden layer values and the new output layer values"""
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values


def choose_action(probability):   # USE THIS ONE!
    random_value = np.random.uniform()
    if random_value < probability:
        # signifies up in openai gym
        return 2
    else:
         # signifies down in openai gym
        return 3
"""
        
def choose_action(probability):
    #random_value = np.random.uniform()
    if 0.5 > probability:
        # signifies up in openai gym
        return 2
    else:
         # signifies down in openai gym
        return 3
"""
def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    """ See here: http://neuralnetworksanddeeplearning.com/chap2.html"""
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def update_weights(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    """ See here: http://sebastianruder.com/optimizing-gradient-descent/index.html#rmsprop"""
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer

def discount_rewards(rewards, gamma):
    """ Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago.
    This implements that logic by discounting the reward on previous actions based on how long ago they were taken"""
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def discount_with_rewards(gradient_log_p, episode_rewards, gamma):
    """ discount the gradient with the normalized rewards """
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards


# In[14]:


from keras import Sequential
def main():
    #env = gym.make("Pong-v0")
    env = gym.make('BreakoutDeterministic-v0')
    observation = env.reset()
    
    episode_number = 0
    reward_sum = 0
    running_reward = None
    prev_processed_observations = None
    
    action_size = env.action_space.n
    
    print(action_size)
    
    input_dimensions = (80, 80)
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                     activation='relu',
                     input_shape=(80,80,1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer = "adam", metrics=['accuracy'])
    
    episode_actions = []
    
    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards = [], [], [], []
    
    train_x_final = []
    
    fitted = False
    
    while True:
        env.render()
        processed_observations, prev_processed_observations = preprocess_observations(observation, prev_processed_observations, input_dimensions)
        #hidden_layer_values, up_probability = apply_neural_nets(processed_observations, weights
        
        #up_probability = happyModel.predict(processed_observations)
        #model.compile(loss=keras.losses.categorical_crossentropy,
        #      optimizer=keras.optimizers.SGD(lr=0.01),
        #      metrics=['accuracy'])
        
        if fitted==True:
            
            test_x = np.array(processed_observations)
            test_x = test_x.reshape((1,) + test_x.shape + (1,))
            up_probability = model.predict(test_x)
            action = choose_action(np.argmax(up_probability))
            #print(action)
            
        episode_observations.append(processed_observations)
        
        action = env.action_space.sample()
        
        # carry out the chosen action
        observation, reward, done, info = env.step(action)
        
        episode_actions.append(action)
        
        reward_sum += reward
        episode_rewards.append(reward)
        #show_state(env)
        #show_state(processed_observations)
        
        # see here: http://cs231n.github.io/neural-networks-2/#losses
        #fake_label = 1 if action == 2 else 0
        #loss_function_gradient = fake_label - up_probability
        #episode_gradient_log_ps.append(loss_function_gradient)
        
        #model.fit((1, np.array(processed_observations), 1), np.array(action))

        if done: # an episode finished
            episode_number += 1
            
            if reward_sum >= 3:
                if fitted == False:
                    train_x = np.array(episode_observations)
                    train_y = np_utils.to_categorical(list(episode_actions))
                    print('shape trainx pre is {0}' .format(train_x.shape))
                    print('shape trainx pre after reshape is {0}' .format(train_x.shape))
                else:
                    train_x = np.concatenate((train_x, episode_observations))
                    train_y = np.concatenate((train_y, np_utils.to_categorical(list(episode_actions))))
                    
                    print('shape trainx post is {0}' .format(train_x.shape))
                    
                train_x_final = train_x.reshape(train_x.shape + (1,))
                model.fit(train_x_final, train_y, epochs=1, batch_size=32, verbose=1)
                
                fitted = True
                
                if episode_number == 500:
                    np.savetxt('train.out', train_x, delimiter=',')
                    
            
            """
            
            if reward_sum >= 2: # only use good training examples 
                if fitted == False:
                    train_x = np.array(episode_observations)
                    train_x = train_x.reshape(train_x.shape + (1,))
                    train_y = np_utils.to_categorical(list(episode_actions))
                    saved_obs = episode_observations
                    saved_actions = episode_actions
                    #print('shape pre is {0}' .format(train_x.shape))
                    #print('shape episode pre is {0}' .format(np.array(episode_observations).shape))
                else:
                    
                    print(reward_sum)
                    print('shape trainx pre is {0}' .format(train_x.shape))
                    # Combine the following values for the episode
                    #print('shape episode post is {0}' .format(np.array(episode_observations).shape))
                    print(np.array(episode_observations).shape)
                    print(np.array(saved_obs).shape)
                    saved_obs.append(episode_observations)
                    saved_actions.append(episode_actions)
                    print(np.array(saved_obs).shape)
                    #print('shape post is {0}' .format(np.array(saved_obs).shape))
                    
                    train_x = np.array(saved_obs)
                    print('shape trainx pre is {0}' .format(train_x.shape))
                    train_x = train_x.reshape(train_x.shape + (1,))
                    print('shape trainx post is {0}' .format(train_x.shape))
                    #episode_rewards = np.array(episode_rewards)
                    
                    #episode_actions = np.array(episode_actions)
                    train_y = np_utils.to_categorical(list(saved_actions))
                      
                
                #if episode_number % 10 == 0:
                
                model.fit(train_x, train_y, epochs=1, batch_size=32, verbose=1)
                
                fitted = True
                """
                
                
                
            episode_observations = []
            episode_actions = []            

            observation = env.reset() # reset env
            running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            print('resetting env. episode reward total was %.2f. running mean: %.2f' % (reward_sum, running_reward))
            reward_sum = 0
            prev_processed_observations = None
            
            


# In[15]:
""" MISSING LOGIC --> Discounting target
 for i, (state_b, action_b, reward_b, next_state_b) in enumerate(minibatch):
            inputs[i:i+1] = state_b
            target = reward_b
            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                target_Q = mainQN.model.predict(next_state_b)[0]
                target = reward_b + gamma * np.amax(mainQN.model.predict(next_state_b)[0])
            targets[i] = mainQN.model.predict(state_b)
            targets[i][action_b] = target
        mainQN.model.fit(inputs, targets, epochs=1, verbose=0)
"""
main()

#https://lilianweng.github.io/lil-log/2018/05/05/implementing-deep-reinforcement-learning-models.html#deep-q-network

# In[ ]:


def HappyModel(input_shape, action_size):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
    ### START CODE HERE ###
    # Feel free to use the suggested outline in the text above to get started, and run through the whole
    # exercise (including the later portions of this notebook) once. The come back also try out other
    # network architectures as well. 
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)
    
    print(X_input.shape)

    # Zero-Padding: pads the border of X_input with zeroes
    #X = ZeroPadding2D((3, 3))(X_input)
    #print(X.shape)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X_input)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(10, activation = 'relu')(X)
    X = Dense(action_size, activation='softmax', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')
    
    
    ### END CODE HERE ###
    
    return model

