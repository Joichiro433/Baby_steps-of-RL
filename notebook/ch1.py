#!/usr/bin/env python
# coding: utf-8

# In[14]:


from typing import *
import os
import random
from enum import Enum
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nptyping import NDArray
from IPython.display import display

# 型推定
from matplotlib.figure import Figure
from matplotlib.axes._subplots import Subplot

sns.set_style('whitegrid')
colors = ['#de3838', '#007bc3', '#ffd12a']
markers = ['o', 'x', ',']

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 100)

cmap = sns.diverging_palette(255, 0, as_cmap=True)  # カラーパレットの定義


# In[17]:


class Action(Enum):
    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class State:
    def __init__(self, row: int = -1, col: int = -1) -> None:
        self.row : int = row
        self.col : int = col
    
    def __repr__(self) -> str:
        return f'<State: [{self.row}, {self.col}]>'

    def __hash__(self) -> int:
        return hash((self.row, self.col))
    
    def __eq__(self, other: object) -> bool:
        return (self.row == other.row) and (self.col == other.col)
    
    def clone(self) -> object:
        return State(row=self.row, col=self.col)


class Environment:
    def __init__(self, grid: List[List[int]], move_prob: float = 0.8) -> None:
        # grid is 2d-array. Its values are treated as an attribute.
        # Kinds of attribute is following.
        #  0: ordinary cell
        #  -1: damage cell (game end)
        #  1: reward cell (game end)
        #  9: block cell (can't locate agent)
        self.grid : List[List[int]] = grid
        self.agent_state : State = State()

        # Default reward is minus. Just like a poison swamp.
        # It means the agent has to reach the goal fast!
        self.default_reward : float = -0.04

        # Agent can move to a selected direction in move_prob.
        # It means the agent will move different direction
        # in (1 - move_prob).
        self.move_prob : float = move_prob
        self.reset()

    @property
    def row_length(self) -> int:
        return len(self.grid)

    @property
    def col_length(self) -> int:
        return len(self.grid[0])
    
    @property
    def actions(self) -> List[Action]:
        """取り得る全てのアクション"""
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self) -> List[State]:
        """取り得る全ての状態"""
        states : List[List[int]] = [State(row=row, col=col) for row in range(self.row_length) for col in range(self.col_length)]
        return states

    def transit_func(self, state: State, action: Action) -> Dict[State, float]:
        transition_probs : Dict[State, float] = {}
        if not self.can_action_at(state=state):
            # Already on the terminal cell.
            return transition_probs
        
        opposite_direction : Action = Action(action.value * -1)

        for a in self.actions:
            prob : float = 0
            if a == action:
                prob = self.move_prob
            elif a != opposite_direction:
                prob = (1 - self.move_prob) / 2
            
            next_state = self._move(state, a)
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob
            
        return transition_probs


    def can_action_at(self, state: State) -> bool:
        if self.grid[state.row][state.col] == 0:
            return True
        return False

    def _move(self, state: State, action: Action) -> State:
        if not self.can_action_at(state=state):
            raise Exception('Can\'t move from here!')
        
        next_state : State = state.clone()

        # Execute an action (move)
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.col -= 1
        elif action == Action.RIGHT:
            next_state.col += 1

        # Check whether a state is out of the grid.
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.col < self.col_length):
            next_state = state

        # Check whether the agent bumped a block cell.
        if self.grid[next_state.row][next_state.col] == 9:
            next_state = state
        
        return next_state

    def reward_func(self, state: State) -> Tuple[float, bool]:
        reward : float = self.default_reward
        done : bool = False

        # Check an attribute of next state.
        attribute : int = self.grid[state.row][state.col]
        if attribute == 1:
            # Get reward! and the game ends.
            reward = 1
            done = True
        if attribute == -1:
            # Get damage! and the game ends.
            reward = -1
            done = True
        
        return reward, done

    def reset(self) -> State:
        # Locate the agent at lower left corner.
        self.agent_state = State(row=self.row_length - 1, col=0)
        return self.agent_state

    def step(self, action: Action) -> Tuple[Optional[State], Optional[float], bool]:
        next_state, reward, done = self.transit(state=self.agent_state, action=action)
        if next_state is not None:
            self.agent_state = next_state
        return next_state, reward, done

    def transit(self, state: State, action: Action) -> Tuple[Optional[State], Optional[float], bool]:
        transition_probs : Dict[State, float] = self.transit_func(state=state, action=action)
        if len(transition_probs) == 0:
            return None, None, True

        next_states : List[State] = []
        probs : List[float] = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])
        
        next_state : State = np.random.choice(next_states, p=probs)
        reward, done = self.reward_func(state=next_state)
        return next_state, reward, done


# In[22]:


class Agent:
    def __init__(self, env: Environment) -> None:
        self.actions : List[Action] = env.actions
    
    def policy(self, state: State) -> Action:
        return random.choice(self.actions)


def main():
    # Make grid environment.
    grid : List[List[int]] = [
        [0, 0, 0, 1],
        [0, 9, 0, -1],
        [0, 0, 0, 0]
    ]
    env : Environment = Environment(grid=grid)
    agent : Agent = Agent(env=env)

    # Try 10 game.
    for i in range(10):
        # Initialize position of agent.
        state : State = env.reset()
        total_reward : float = 0
        done : bool = False

        while not done:
            action : Action = agent.policy(state=state)
            next_state, reward, done = env.step(action=action)
            total_reward += reward
            state = next_state

        print(f'Episode {i}: Agent gets {total_reward:.2f} reward.')


main()


# In[ ]:




