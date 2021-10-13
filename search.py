# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

"""
    Name: Kai Ying Chan
    Date: 10/12/2021
    Description: This is a file that uses DFS, BFS, UCS, A* search to find paths for pacman to move
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    # node is initial state, actions, cost
    startingNode = (problem.getStartState(), [], 0)

    # if start is goal return nothing
    if problem.isGoalState(problem.getStartState()):
        return []

    # create LIFO Queue with node as only element
    frontier = util.Stack()
    frontier.push(startingNode)

    # create list for reached Nodes
    reachedNodes = []

    # loop until frontier is empty
    while not frontier.isEmpty():
        # choose the shallowest node in frontier
        currentState, actions, currentCost = frontier.pop()

        # add current state to reachedNodes if current state hasn't been expanded
        if currentState not in reachedNodes:
            reachedNodes.append(currentState)

            # return actions if finds goal
            if problem.isGoalState(currentState):
                return actions
            else:
                # expand successor and add to frontier
                for successorState, successorAction, successorCost in problem.getSuccessors(currentState):
                    child = (successorState, actions + [successorAction], currentCost + successorCost)
                    frontier.push(child)

    return actions

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    # node is initial state, actions, cost
    startingNode = (problem.getStartState(), [], 0)

    # if start is goal return nothing
    if problem.isGoalState(problem.getStartState()):
        return []

    # create FIFO Queue with node as only element
    frontier = util.Queue()
    frontier.push(startingNode)

    # create list for reached Nodes
    reachedNodes = []

    # loop until frontier is empty
    while not frontier.isEmpty():
        # choose the shallowest node in frontier
        currentState, actions, currentCost = frontier.pop()

        # add current state to reachedNodes if current state hasn't been expanded
        if currentState not in reachedNodes:
            reachedNodes.append(currentState)

            # return actions if finds goal
            if problem.isGoalState(currentState):
                return actions
            else:
                # expand successor and add to frontier
                for successorState, successorAction, successorCost in problem.getSuccessors(currentState):
                    child = (successorState, actions + [successorAction], currentCost + successorCost)
                    frontier.push(child)

    return actions

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    # node is initial state, actions, cost
    startingNode = (problem.getStartState(), [], 0)

    # if start is goal return nothing
    if problem.isGoalState(problem.getStartState()):
        return []

    # create FIFO Priority Queue with node as only element ad priority of 0
    frontier = util.PriorityQueue()
    frontier.push(startingNode, 0)

    # create dictionary for reached Nodes {'State',Cost}
    reachedNodes = {}

    # loop until frontier is empty
    while not frontier.isEmpty():
        # choose the shallowest node in frontier
        currentState, actions, currentCost = frontier.pop()

        # add current state with the corresponding cost to reachedNodes dictionary
        # if current state has not been reached or current cost is less than previous cost
        if (currentState not in reachedNodes) or currentCost < reachedNodes[currentState]:
            reachedNodes[currentState] = currentCost

            # return actions if finds goal
            if problem.isGoalState(currentState):
                return actions
            else:
                # expand successor and add to frontier
                for successorState, successorAction, successorCost in problem.getSuccessors(currentState):
                    child = (successorState, actions + [successorAction], currentCost + successorCost)
                    frontier.push(child, currentCost + successorCost)

    return actions

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    # node is initial state, actions, cost
    startingNode = (problem.getStartState(), [], 0)

    # if start is goal return nothing
    if problem.isGoalState(problem.getStartState()):
        return []

    # create FIFO Priority Queue with node as only element ad priority of 0
    frontier = util.PriorityQueue()
    frontier.push(startingNode, 0)

    # create list for reached Nodes [state, cost]
    reachedNodes = {}

    # loop until frontier is empty
    while not frontier.isEmpty():
        # choose the shallowest node in frontier
        currentState, actions, currentCost = frontier.pop()

        # add current state with the corresponding cost to reachedNodes dictionary
        # if current state has not been reached or current cost is less than previous cost
        if (currentState not in reachedNodes) or currentCost < reachedNodes[currentState]:
            reachedNodes[currentState] = currentCost

            # return actions if finds goal
            if problem.isGoalState(currentState):
                return actions
            else:
                # expand successor and add to frontier
                for successorState, successorAction, successorCost in problem.getSuccessors(currentState):
                    child = (successorState, actions + [successorAction], currentCost + successorCost)

                    # check if child node has been reached
                    childIsReached = (child[0] in reachedNodes) and ((currentCost+successorCost) >= reachedNodes.get(child[0]))

                    if not childIsReached:
                        heuristicValue = heuristic(successorState, problem)
                        frontier.update(child, heuristicValue + currentCost + successorCost)

    return actions

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
