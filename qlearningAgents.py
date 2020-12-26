# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discount (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  ### Get policy and get actions are the same functions. Get policy will return the optimal policy but get action will return based on the 
  ### Epsilon value.
  def __init__(self, **args):
    ReinforcementAgent.__init__(self, **args)
    self.qvalues = util.Counter()

  def getQValue(self, state, action):
    ### Here Q value must be a combination of state and action. This will increase the size of states by 4, because there are 4 actions., north,east,west and south 
    return self.qvalues[(state,action)]
    ### qvalues will be a combination of state and actions.


  def computeValueFromQValues(self, state):
    return self.getValue(state)

  def computeActionFromQValues(self, state):
    return self.getPolicy(state)

  def getAction(self, state):
    ### Question no 5 Code
    ### Here we will use one utility function to choose between two types of actions. The random action and the best action.
    if util.flipCoin(self.epsilon):
      return random.choice(self.getLegalActions(state))
    else:
      return self.getPolicy(state)



  def getPolicy(self, state):
    ###Policy will be the best action for the time being.
    bestaction = None
    bestvalue = 0
    for action in self.getLegalActions(state):
      value = self.getQValue(state, action)
      if value > bestvalue or bestaction is None:
        bestvalue = value
        bestaction = action
    return bestaction


  def getValue(self, state):
    ### Find the max value from each legal action.
    ### Choose maximum value from the legal action values.
    oldqvalues = []
    for action in self.getLegalActions(state):
      oldqvalues.append(self.getQValue(state, action))
    if len(self.getLegalActions(state)) == 0:
      return 0.0
    else:
      return max(oldqvalues)
  def update(self, state, action, nextState, reward):
    ### Simple update from slides.
    noaction = (1 - self.alpha) * self.getQValue(state, action)
    if len(self.getLegalActions(nextState)) == 0:
      sample = reward
    else:
      sample = reward + (self.discount * max([self.getQValue(nextState, next_action) for next_action in self.getLegalActions(nextState)]))
    withaction = self.alpha * sample
    self.qvalues[(state, action)] = noaction + withaction    


 

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    ### Perform specific action for the current state and return that action.
    self.doAction(state,QLearningAgent.getAction(self,state))
    return QLearningAgent.getAction(self,state)


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)
    self.weights = util.Counter()
    
  def getWeights(self):
    return self.weights  
 
 

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    "*** YOUR CODE HERE ***"
    qvalue = 0
    features = self.featExtractor.getFeatures(state, action)
    #Each feature is in the form of dictionary {((3, 3), 'east'): 1.0}. Each key is a combination of coordinate and direction. Each value represents the old qvalue.
    for feature in features.keys():
      qvalue += features[feature] * self.weights[feature]
    return qvalue

  def update(self, state, action, nextState, reward):
    ### Question no 9 Code.
    qvalue = self.getQValue(state,action)
    features = self.featExtractor.getFeatures(state,action)
    qvaluenext = self.getValue(nextState)
    for feature in features.keys():
      ### Same function as that of given in the assignemt.
      ### We will update the weights for the current feature.
      ### Correction Factor = alpha*((reward + discount*(qvaluenext))- qvalue)
      self.weights[feature] =  self.weights[feature] + self.alpha*((reward + self.discount*(qvaluenext))- qvalue)*features[feature]








  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      "*** YOUR CODE HERE ***"
      ### if training is finished then we have to exit.
      exit(0)

