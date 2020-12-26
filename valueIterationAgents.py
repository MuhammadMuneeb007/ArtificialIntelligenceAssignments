# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.
    
      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)

    
    """
    ### MDP Class provides us these functionality which can be used to Implement the ValueIteration function for Pacman and GridVersion.
    '''    
    def getStates(self):
    def getStartState(self):  
    def getPossibleActions(self, state):      
    def getTransitionStatesAndProbs(self, state, action):
    def getReward(self, state, action, nextState):  
    def isTerminal(self, state):
    '''
    
    self.mdp = mdp
    self.discount = discount         # Discount factor
    self.iterations = iterations     # Number of Iterations
    self.values = util.Counter()     # A Counter is a dict with default 0
    self.oldvalues = util.Counter()

    for loop in range(0,self.iterations):
      self.oldvalues =  self.values.copy()
      for state in self.mdp.getStates():                   # For each state we have to perform specific functions. 
        actionvalue = []
        if self.mdp.isTerminal(state):                     # If state is terminal then Value should be 0.
            self.values[state] = 0
        else:
            for action in self.mdp.getPossibleActions(state): 
              value = 0
              for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                value = value + transition[1]*(self.mdp.getReward(state, action, transition[0]) + self.discount * self.oldvalues[transition[0]])
              actionvalue.append(value)
            self.values[state] = max(actionvalue)
      
  def getValue(self, state):
    return self.values[state]
  
  def computeActionFromValues(self,state):
    return self.getAction(state)  
    
  def computeQValueFromValues(self,state, action):
    return self.getQValue(state,action)

  def getQValue(self, state, action):
    value = 0
    
    ### Each action have transitions to multiple states.
    for transition in self.mdp.getTransitionStatesAndProbs(state, action):
      value = value + transition[1]*(self.mdp.getReward(state, action, transition[0]) + self.discount * self.values[transition[0]])
      ### For each action we have to find the transition probabilities to other states. 
    return value
    
  def getPolicy(self, state):
    return self.getAction(state)  
    
    



  def getAction(self, state):
    if self.mdp.isTerminal(state):      ### Terminal state have no actions.
      return None
    else:
      bestvalue = -1000000000
      bestaction = None
      for action in self.mdp.getPossibleActions(state): 
        value = self.getQValue(state,action)  ### At each state we will find the maximum of the qvalue and it will decide the function.
        if value > bestvalue:
          bestaction = action
          bestvalue = value
      return bestaction    
  
