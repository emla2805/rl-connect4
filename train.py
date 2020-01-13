import tensorflow as tf
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver

from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from game import Connect4Environment


def eval_against_random_policy(environment, policy, num_episodes=10):
    random_policy = random_tf_policy.RandomTFPolicy(environment.time_step_spec(),
                                                    environment.action_spec())
    policies = [policy, random_policy]
    wins = [0.0, 0.0]
    for _ in range(num_episodes):
        time_step = environment.reset()
        player = 0

        while not time_step.is_last():
            action_step = policies[player].action(time_step)
            time_step = environment.step(action_step.action)
            wins[player] += time_step.reward.numpy()[0]
            player = 1 - player

    return wins


if __name__ == "__main__":
    # train_py_env = Connect4Environment()
    eval_py_env = Connect4Environment()
    # train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    train_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(
            [Connect4Environment] * 16
        )
    )

    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    collect_steps_per_iteration = 100
    replay_buffer_capacity = 100000

    fc_layer_params = (100,)

    batch_size = 64
    learning_rate = 1e-3
    log_interval = 5

    num_eval_episodes = 10
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

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        train_env,
        agent.collect_policy,
        observers=[replay_buffer.add_batch],
        num_steps=collect_steps_per_iteration,
    )

    # Initial data collection
    collect_driver.run()

    # Dataset generates trajectories with shape [BxTx...] where
    # T = n_step_update + 1.
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=batch_size, num_steps=2
    ).prefetch(3)

    iterator = iter(dataset)

    agent.train = common.function(agent.train)

    def train_one_iteration():

        # Collect a few steps using collect_policy and save to the replay buffer.
        for _ in range(collect_steps_per_iteration):
            collect_driver.run()

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(iterator)
        train_loss = agent.train(experience)

        iteration = agent.train_step_counter.numpy()
        print("iteration: {0} loss: {1}".format(iteration, train_loss.loss))

    num_iterations = 50
    for _ in range(num_iterations):
        train_one_iteration()
        wins = eval_against_random_policy(eval_env, agent.policy, num_episodes=100)
        print(wins)
