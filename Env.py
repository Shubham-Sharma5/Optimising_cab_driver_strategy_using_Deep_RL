# Import routines

import numpy as np
import math
import random
import itertools
# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = list(itertools.permutations(range(0,5), 2)) +[(0,0)] 
        self.state_space = list(itertools.product(range(0,5), range(0,24), range(0,7)))
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encode = np.zeros((m+t+d,))
        state_encode[state[0]] = 1
        state_encode[m+state[1]] = 1
        state_encode[m+t+state[2]] = 1
        return state_encode


    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2) 
        elif location == 1:
            requests = np.random.poisson(12)
        elif location == 2:
            requests = np.random.poisson(4)
        elif location == 3:
            requests = np.random.poisson(7)
        elif location == 4:
            requests = np.random.poisson(8)

        if requests >15: #Setting an upper bound of requests
            requests =15

        possible_actions_index = random.sample(range(0, (m-1)*m), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        actions.append((0,0))
        possible_actions_index.append(20)
        return possible_actions_index,actions   



    def reward_and_time_spent_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        cur_loc, cur_time, cur_day = state
        reward = 0
        total_time = 0 #Time duration between current and drop location
        #Case in which the driver goes offline
        if action == (0,0):
            reward -= C #Cost of fuel consumed per hour will be deducted from total reward
            total_time = 1
        #Case in which the driver is accepting requests
        else:
            pickup_loc, drop_loc = action
            if pickup_loc != cur_loc: #If pickup and current locations are different
                time_to_pickup_loc = int(Time_matrix[cur_loc, pickup_loc, cur_time, cur_day])
                reward -= (C*time_to_pickup_loc)
                total_time += time_to_pickup_loc
                time_at_pickup = (cur_time+time_to_pickup_loc)%24 #Time by which the driver reaches the pickup location
                #Current day is changed to next day if the given condition is true
                if time_at_pickup < cur_time: 
                    cur_day = (cur_day+1)%7 
                cur_time = time_at_pickup

            time_bet_pickup_drop = int(Time_matrix[pickup_loc][drop_loc][cur_time][cur_day]) #Time between pickup and drop location
            reward += ((R-C)*time_bet_pickup_drop) #Reward for the ride duration
            total_time += time_bet_pickup_drop
        return reward, total_time



    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        cur_loc, cur_time, cur_day = state #Getting the current values of the state
        #Case when the driver went offline
        if action == (0,0): 
            cur_time = (cur_time+1)%24 #Increase the cur_time by one hour
            if cur_time == 0: #Updating the day in case of day change
                cur_day = (cur_day+1)%7 

            next_state = (cur_loc, cur_time, cur_day) #Storing the values of the next state in a tuple
        #Case when the driver is accepting requests
        else:
            pickup_loc, drop_loc = action #Getting the pickup and drop location
            time_at_pickup = 0
            #Case when the pickup is from a different location than the current one 
            if cur_loc != pickup_loc:
                time_to_pickup_loc = int(Time_matrix[cur_loc][pickup_loc][cur_time][cur_day])

                time_at_pickup = (cur_time+time_to_pickup_loc)%24 #Time by which the driver reaches the pickup location
                #Current day is changed to next day if the given condition is true
                if time_at_pickup < cur_time: 
                    cur_day = (cur_day+1)%7
                cur_time = time_at_pickup #Current time is updated as the time at pickup
            time_to_drop_from_pickup = int(Time_matrix[pickup_loc][drop_loc][cur_time][cur_day]) #Duration between pickup and drop loc

            time_at_drop = (cur_time+time_to_drop_from_pickup)%24 #Time by which the cab reaches the drop location
            #Current day is updated to next day in case the drop was completed after midnight
            if time_at_drop < cur_time:
                cur_day = (cur_day+1)%7
            cur_time = time_at_drop #Setting cur_time as the time by which drop was made
            next_state = (drop_loc, cur_time, cur_day) #Tuple of the next state
        
        return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
