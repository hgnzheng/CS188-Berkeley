# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        # Sanity check in case no more food on the game board
        if len(newFood.asList()) == 0:
            foodScore = 0
        else:
            # Use manhattan distance to find the closest food distance
            closestFoodDist = min([manhattanDistance(newPos, foodPos) 
                for foodPos in newFood.asList()])
            # Take the reciprocal as a part of the evaluation function
            foodScore = 1 / closestFoodDist
        
        # Use manhattan distance to find the closest ghost distance
        closestGhostDist = min([manhattanDistance(newPos, 
            ghostState.configuration.pos) for ghostState in newGhostStates])
        # Sanity check in case there is no more ghost
        if closestGhostDist == 0:
            ghostScore = 0
        else:
            # Take the reciprocal as a part of the evaluation function
            ghostScore = 2 / closestGhostDist
        return successorGameState.getScore() + foodScore - ghostScore

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Sanity check to see if we reach the end of the game/condition
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        # Initialize the maxValue
        maxValue = -float('inf')
        # Initialize resulting move
        result = None
        # Iterate through all possible actions and compare the value
        for action in gameState.getLegalActions(agentIndex=0):
            # Extract the successor of the current gameState
            successor = gameState.generateSuccessor(agentIndex=0, action=action)
            # Temporary variable used to compare with maxValue
            temp_val = self.min_value(successor, 1, 0)
            # Compare values and update if necessary
            if temp_val > maxValue:
                maxValue = temp_val
                result = action
        return result
    
    def max_value(self, gameState, agentIndex, depth):
        """
        Returns the maximum value obtained from iterating through the
        given gameState.
        """
        # Base case when we reach the end of the game/condition
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        # Same as pseudocode in lecture, initialize the maxValue
        maxValue = -float('inf')
        # Get all possible actions from the current gameState
        legalAction = gameState.getLegalActions(agentIndex=agentIndex)
        # Iterate through all possible actions to find the max
        for action in legalAction:
            # Extract the successor of the current gameState
            successor = gameState.generateSuccessor(agentIndex, action)
            # Update maxValue when necessary
            maxValue = max(maxValue, self.min_value(successor, agentIndex + 1, depth))
        return maxValue
    
    def min_value(self, gameState, agentIndex, depth):
        """
        Returns the minimum value obtained from iterating through the
        given gameState.
        """
        # Base case when we reach the end of the game/condition
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        # Same as pseudocode in lecture, initialize the maxValue
        minValue = float('inf')
        # Get all possible actions from the current gameState
        legalAction = gameState.getLegalActions(agentIndex=agentIndex)
        # Iterate through all possible actions to find the max
        for action in legalAction:
            # Extract the successor of the current gameState
            successor = gameState.generateSuccessor(agentIndex, action)
            # Update maxValue when necessary; increment depth when we have 
            # iterated through all agents; otherwise update agent index
            minValue = min(minValue, self.max_value(successor, 0, depth + 1) 
                if agentIndex == gameState.getNumAgents() - 1 
                else self.min_value(successor, agentIndex + 1, depth))
        return minValue

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Initialize variables used for comparison later
        alpha = -float('inf')
        beta = float('inf')
        maxValue = -float('inf')
        # Initialize the resulting move
        result = None
        # Iterate through all possible actions and compare the value
        for action in gameState.getLegalActions(agentIndex=0):
            successor = gameState.generateSuccessor(agentIndex=0, action=action)
            temp_val = self.min_value(successor, agentIndex=1, depth=0, alpha=alpha, beta=beta)
            if temp_val > maxValue:
                maxValue = temp_val
                result = action
            if maxValue >= beta:
                return action
            alpha = max(alpha, maxValue)
        return result

    def max_value(self, gameState, agentIndex, depth, alpha, beta):
        """
        Returns the maximum value obtained from iterating through the
        given gameState.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        maxValue = -float('inf')
        legalAction = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalAction:
            successor = gameState.generateSuccessor(agentIndex, action)
            maxValue = max(maxValue, self.min_value(successor, agentIndex + 1, depth, alpha, beta))
            if maxValue > beta:
                return maxValue
            alpha = max(alpha, maxValue)
        return maxValue
    
    def min_value(self, gameState, agentIndex, depth, alpha, beta):
        """
        Returns the minimum value obtained from iterating through the
        given gameState.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        minValue = float('inf')
        legalAction = gameState.getLegalActions(agentIndex=agentIndex)
        for action in legalAction:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                minValue = min(minValue, self.max_value(successor, 0, depth + 1, alpha, beta))
            else:
                minValue = min(minValue, self.min_value(successor, agentIndex + 1, depth, alpha, beta))
            if minValue < alpha:
                return minValue
            beta = min(beta, minValue)
        return minValue

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        maxValue = -float('inf')
        result = None
        for action in gameState.getLegalActions(agentIndex=0):
            successor = gameState.generateSuccessor(agentIndex=0, action=action)
            temp_val = self.exp_value(successor, 1, 0)
            if temp_val > maxValue:
                maxValue = temp_val
                result = action
        return result

    def max_value(self, gameState, agentIndex, depth):
        """
        Returns the maximum value obtained from iterating through the
        given gameState.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        maxValue = -float('inf')
        legalAction = gameState.getLegalActions(agentIndex=0)
        for action in legalAction:
            successor = gameState.generateSuccessor(0, action)
            maxValue = max(maxValue, self.exp_value(successor, 1, depth))
        return maxValue
    
    def exp_value(self, gameState, agentIndex, depth):
        """
        Returns the minimum value obtained from iterating through the
        given gameState.
        """
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        exp = 0
        legalActions = gameState.getLegalActions(agentIndex=agentIndex)
        prob = 1.0 / len(legalActions)
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            if agentIndex == gameState.getNumAgents() - 1:
                exp_temp = self.max_value(successor, 0, depth + 1)
            else:
                exp_temp = self.exp_value(successor, agentIndex + 1, depth)
            exp += prob * exp_temp
        return exp

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    # Useful information you can extract from a GameState (pacman.py)
    currPos = currentGameState.getPacmanPosition()
    currFood = currentGameState.getFood()
    currGhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in currGhostStates]

    # Check if there are still food remaining
    if len(currFood.asList()) == 0:
        foodScore = 0
    else:
        # Find the closest food using manhattan distance
        closestFoodDist = min([manhattanDistance(currPos, foodPos) 
            for foodPos in currFood.asList()])
        # Feature score by taking the reciprocal of closest food distance
        foodScore = 1 / closestFoodDist
    
    # Use manhattan distance to find the closest ghost distance
    closestGhostDist = min([manhattanDistance(currPos, 
        ghostState.configuration.pos) for ghostState in currGhostStates])
    # Sanity check when no ghost exist
    if closestGhostDist == 0:
        ghostScore = 0
    else:
        # Take the reciprocal in the final evaluation function
        ghostScore = 2 / closestGhostDist
    return currentGameState.getScore() + foodScore - ghostScore + sum(ScaredTimes)

# Abbreviation
better = betterEvaluationFunction
