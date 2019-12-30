import math
import gym
import numpy as np

class TradeEnv():
    """
    This class is the trading environment (render) of our project. 

    The trading agent calls the class by giving an action at the time t. 
    Then the render gives back the new portfolio at the next step (time t+1). 

    # Parameters:
    - windonw_length: number of time inputs looked in the past to build the input tensor
    - portfolio_value: initial value of the portfolio 
    - trading_cost: cost (in % of the traded stocks) the agent will pay to execute the action 
    - interest_rate: rate (in % of the money the agent has) the agent will either get at each step 
                     if he has a positive amount of money or pay if he has a negative amount of money
    -train_size: fraction of data used for the training of the agent, (train -> | time T | -> test)
    """

    def __init__(self, path = './input.npy', window_length=50,
                 portfolio_value= 10000, trading_cost= 0.25/100,interest_rate= 0.02/250, train_size = 0.7):
        
        self.path = path                                # path to numpy data
        self.data = np.load(self.path)                  # load the input tensor
        self.portfolio_value = portfolio_value          # initial input value
        self.window_length = window_length              # window of previous samples
        self.trading_cost = trading_cost                # trading costs
        self.interest_rate = interest_rate              # interest rate on money
        self.nb_features = self.data.shape[0]           # number of features
        self.nb_stocks = self.data.shape[1]             # number of stocks
        self.nb_samples = self.data.shape[2]            # number of samples
        self.end_train = int((self.nb_samples-self.window_length)*train_size)    # number of training samples
        self.index = None                               # initial index - integer / time step t currently happening
        self.state = None                               # initial state - tuple / data, portfolio weights, portfolio value
        self.seed()                                     # initial seed
        self.done = False                               # epoch indicator


    def return_pf(self):
        return self.portfolio_value
        
    def readTensor(self,X,t):
        # Input tensor for NN. All features, All stocks, Current time up to window previous values
        return X[ : , :, t-self.window_length:t ]
    
    def readUpdate(self, t):
        # Return of each stock for the period t 
        return np.array([1+self.interest_rate]+self.data[-1,:,t].tolist())

    def seed(self, seed=None):
        # Set a random seed for reproducibility
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    
    def reset(self, w_init, p_init, t=0 ):
        # Restart the environments' epoch with given first window of data, initial portfolio weights and value of portfolio
        self.state = (self.readTensor(self.data, self.window_length), w_init , p_init )
        self.index = self.window_length + t
        self.done = False
        return self.state, self.done
  
    def step(self, action):
        """
        Main function of the render.

        At each step t, the trading agent gives (the action he wants to do) the new value of the weights of the portfolio. 
        The function computes the new value of the portfolio at the step (t+1), it returns also the reward associated with the action the agent took. 
        The reward is defined as the evolution of the the value of the portfolio in %. 
        """
        index = self.index                              # current time step
        data = self.readTensor(self.data, index)        # current input tensor
        done = self.done                                # current epoch indicator

        # Beginning of the day / period
        state = self.state                              # state space at the beginning of the period
        w_previous = state[1]                           # weights of the portfolio at the beginning of the day
        pf_previous = state[2]                          # value of portfolio at the beginning of the day        
        update_vector = self.readUpdate(index)          # vector of opening price of the period divided by opening price of previous period
        w_alloc = action                                # action - chosen weights for portfolio for the next step        
        cost = pf_previous * np.linalg.norm((w_alloc-w_previous),ord=1) * self.trading_cost    # compute transaction cost
        v_alloc = pf_previous * w_alloc                 # convert weight vector into value vector
        v_trans = v_alloc - np.array([cost] + [0]*self.nb_stocks) # substract the transaction cost from each value vector
        
        # End of the day / period
        v_evol = v_trans*update_vector                  # compute value evolution of portfolio 
        pf_evol = np.sum(v_evol)                        # compute the total new portfolio value
        w_evol = v_evol/pf_evol                         # compute weight vector of portfolio
        reward = (pf_evol-pf_previous)/pf_previous      # compute instanteanous reward
        index = index + 1                               # update index
        state = (self.readTensor(self.data, index), w_evol, pf_evol) # compute new state

        if index >= self.end_train: done = True         # check if epoch has ended
        self.state = state                              # save state
        self.index = index                              # save time step
        self.done = done                                # save epoch indicato
        
        return state, reward, done
        
        
        
        
        
        
 
