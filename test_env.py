import numpy as np
import tensorflow as tf
from game_tf import Connect4Environment


if __name__ == "__main__":
    tf_env = Connect4Environment()
    time_step = tf_env.reset()
    rewards = []
    steps = []
    num_episodes = 100
    outcomes = {
        -1: 0,
        0: 0,
        1: 0,
    }

    for _ in range(num_episodes):
        episode_reward = 0
        episode_steps = 0
        while not time_step.is_last():
            legal_actions = tf_env.legal_actions()
            max_index = tf.cast(tf.shape(legal_actions)[0], tf.int64)
            action_index = tf.random.uniform([], 0, max_index, dtype=tf.int64)
            action = legal_actions[action_index]
            time_step = tf_env.step(action)
            episode_steps += 1
            episode_reward += time_step.reward.numpy()

        outcomes[tf_env.winner.numpy()] += 1

        rewards.append(episode_reward)
        steps.append(episode_steps)
        time_step = tf_env.reset()

    num_steps = np.sum(steps)
    avg_length = np.mean(steps)
    avg_reward = np.mean(rewards)

    print("num_episodes:", num_episodes, "num_steps:", num_steps)
    print("avg_length", avg_length, "avg_reward:", avg_reward)
    print(outcomes)
