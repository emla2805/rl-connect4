import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent

from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from game import Connect4Environment


def eval_against_random_policy(env, policy, num_episodes=10):
    random_policy = random_tf_policy.RandomTFPolicy(
        env.time_step_spec(), env.action_spec()
    )
    wins = [0.0, 0.0]
    for player_pos in range(2):
        if player_pos == 0:
            policies = [policy, random_policy]
        else:
            policies = [random_policy, policy]

        for _ in range(num_episodes):
            time_step = env.reset()
            player = 0

            while not time_step.is_last():
                action_step = policies[player].action(time_step)
                time_step = env.step(action_step.action)
                if player_pos == 0:
                    wins[player] += time_step.reward.numpy()[0]
                else:
                    wins[1 - player] += time_step.reward.numpy()[0]
                player = 1 - player

    return wins


# @tf.function
def collect_step(env, policy, buffer):
    time_step = env.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = env.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)


if __name__ == "__main__":
    train_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            [Connect4Environment] * 16
        )
    )
    eval_py_env = Connect4Environment()
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    num_iterations = 10000
    collect_steps_per_iteration = 100
    replay_buffer_capacity = 100_000

    fc_layer_params = (100,)

    batch_size = 64
    learning_rate = 1e-3
    log_interval = 200

    eval_interval = 1000

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params,
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=global_step,
    )
    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_capacity,
    )

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)

    iterator = iter(dataset)

    agent.train = common.function(agent.train)

    random_policy = random_tf_policy.RandomTFPolicy(
        train_env.time_step_spec(), train_env.action_spec()
    )

    # Collect initial data
    for _ in range(200):
        collect_step(train_env, random_policy, replay_buffer)

    for _ in range(num_iterations):

        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()
        if step % log_interval == 0:
            print(f"Step: {step}, Loss: {train_loss}")

        if step % eval_interval == 0:
            wins = eval_against_random_policy(
                eval_env, agent.policy, num_episodes=100
            )
            print(f"Step: {step}, [Wins, Losses]: {wins}")
