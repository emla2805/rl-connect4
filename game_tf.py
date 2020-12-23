import tensorflow as tf

from tf_agents import specs
from tf_agents.environments import tf_environment, utils
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import common

FIRST = ts.StepType.FIRST
MID = ts.StepType.MID
LAST = ts.StepType.LAST


def conv2d(input, filter):
    input = tf.reshape(input, [1] + input.shape + [1])
    return tf.nn.conv2d(input, filter, strides=1, padding="VALID")


class Connect4Environment(tf_environment.TFEnvironment):
    def __init__(self):
        self._initial_state = tf.zeros((6, 7), dtype=tf.int32)
        action_spec = specs.BoundedTensorSpec(
            [], tf.int32, minimum=0, maximum=7 - 1, name="action"
        )
        observation_spec = specs.BoundedTensorSpec(
            [], tf.int32, minimum=-1, maximum=1, name="observation"
        )
        time_step_spec = ts.time_step_spec(observation_spec)
        super(Connect4Environment, self).__init__(time_step_spec, action_spec)
        self._state = common.create_variable(
            "state", self._initial_state, dtype=tf.int32
        )
        self.current_player = common.create_variable("current_player", 1, dtype=tf.int32)
        self.winner = common.create_variable("winner", 0, dtype=tf.int32)
        self.episode_ended = common.create_variable("episode_ended", False, dtype=tf.bool)
        self.steps = common.create_variable("steps", 0)

    def _current_time_step(self):
        if self.steps == 0:
            step_type, reward, discount = (
                tf.constant(FIRST, dtype=tf.int32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(1.0, dtype=tf.float32),
            )
        elif self.episode_ended and self.winner != 0:
            step_type, reward, discount = (
                tf.constant(LAST, dtype=tf.int32),
                tf.constant(1.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
            )
        elif self.episode_ended:
            step_type, reward, discount = (
                tf.constant(LAST, dtype=tf.int32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(0.0, dtype=tf.float32),
            )
        else:
            step_type, reward, discount = (
                tf.constant(MID, dtype=tf.int32),
                tf.constant(0.0, dtype=tf.float32),
                tf.constant(1.0, dtype=tf.float32),
            )

        return ts.TimeStep(step_type, reward, discount, self._state)

    def _reset(self):
        self._state.assign(self._initial_state)
        self.current_player.assign(1)
        self.episode_ended.assign(False)
        self.winner.assign(0)
        self.steps.assign(0)
        return self.current_time_step()

    def _step(self, action):
        if self.episode_ended:
            return self.reset()

        self.make_move(action)

        if self.connected_four():
            self.episode_ended.assign(True)
            self.winner.assign(self.current_player)

        if self.board_full():
            self.episode_ended.assign(True)

        self.steps.assign_add(1)
        self.current_player.assign(self.current_player * -1)

        return self.current_time_step()

    def connected_four(self):
        target = 4 * self.current_player
        # Horizontal
        col = tf.ones((4, 1, 1, 1), dtype=tf.int32)
        if tf.reduce_any(conv2d(self._state, col) == target):
            return True

        # Vertical
        row = tf.ones((1, 4, 1, 1), dtype=tf.int32)
        if tf.reduce_any(conv2d(self._state, row) == target):
            return True

        # Diagonal \
        diag = tf.reshape(tf.eye(4, dtype=tf.int32), [4, 4, 1, 1])
        if tf.reduce_any(conv2d(self._state, diag) == target):
            return True

        # Diagonal /
        diag2 = tf.reverse(diag, axis=[0])
        if tf.reduce_any(conv2d(self._state, diag2) == target):
            return True

        return False

    def make_move(self, action):
        row = tf.where(self._state[:, action] == 0)[-1][0]
        self._state[row, action].assign(self.current_player)

    def legal_actions(self):
        return tf.where(self._state[0, :] == 0)[:, 0]

    def board_full(self):
        return tf.reduce_all(self._state != 0)

