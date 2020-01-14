import numpy as np
from numpy.lib.stride_tricks import as_strided

from tf_agents.environments import py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts


def conv2d(a, f):
    s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
    subM = as_strided(a, shape=s, strides=a.strides * 2)
    return np.einsum("ij,ijkl->kl", f, subM)


class Connect4Environment(py_environment.PyEnvironment):
    def __init__(self):
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype="int32", minimum=0, maximum=7 - 1, name="action"
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2, 6, 7),
            dtype="int32",
            minimum=0,
            maximum=1,
            name="observation",
        )
        self._state = np.zeros((2, 6, 7), dtype=np.int32)
        self._current_player = 0
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = np.zeros((2, 6, 7), dtype=np.int32)
        self._current_player = 0
        self._episode_ended = False
        return ts.restart(self._state)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        if action in self.legal_actions():
            self.make_move(action)

        if self.connected_four(player=0):
            self._episode_ended = True
            self._current_player = 1 - self._current_player
            return ts.termination(self._state, reward=1.0)
        elif self.board_full():  # Draw
            self._episode_ended = True
            self._current_player = 1 - self._current_player
            return ts.termination(self._state, reward=0.0)
        else:
            self._current_player = 1 - self._current_player
            return ts.transition(self._state, reward=0.0)

    def connected_four(self, player):
        # Horizontal
        col = np.ones((4, 1), dtype=int)
        if np.any(conv2d(self._state[player], col) == 4):
            return True

        # Vertical
        row = np.ones((1, 4), dtype=int)
        if np.any(conv2d(self._state[player], row) == 4):
            return True

        # Diagonal \
        diag = np.eye(4, dtype=int)
        if np.any(conv2d(self._state[player], diag) == 4):
            return True

        # Diagonal /
        diag2 = np.fliplr(diag)
        if np.any(conv2d(self._state[player], diag2) == 4):
            return True

        return False

    def legal_actions(self):
        dense_state = np.max(self._state, axis=0)
        return np.where(dense_state[0, :] == 0)[0]

    def make_move(self, action):
        dense_state = np.max(self._state, axis=0)
        free_rows = np.where(dense_state[:, action] == 0)[0]

        self._state[self._current_player, free_rows[-1], action] = 1

    def board_full(self):
        dense_state = np.max(self._state, axis=0)
        return np.all(dense_state == 1)

    def render(self, mode="human"):
        print(f"\nRound: {self._turn}")
        b = self._state.copy()
        b[1][b[1] == 1] = 2
        b = np.max(b, axis=0)
        d = {0: " ", 1: "X", 2: "O"}

        for row in b:
            print("\t", end="")
            for cell in row:
                print(f"| {d[cell]}", end=" ")
            print("|")

        print("\t  _   _   _   _   _   _   _ ")
        print("\t  1   2   3   4   5   6   7 ")


if __name__ == "__main__":
    env = Connect4Environment()
    utils.validate_py_environment(env, episodes=10)
