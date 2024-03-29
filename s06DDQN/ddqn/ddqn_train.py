import gym
import numpy as np
import argparse
from s06DDQN.ddqn.DDQN import DDQN
from s06DDQN.ddqn.utils import plot_learning_curve, create_directory

parser = argparse.ArgumentParser()
parser.add_argument('--max_episodes', type=int, default=20)
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/DDQN/')
parser.add_argument('--reward_path', type=str, default='./output_images/avg_reward.png')
parser.add_argument('--epsilon_path', type=str, default='./output_images/epsilon.png')

args = parser.parse_args()


def main():
    env = gym.make('LunarLander-v2')
    agent = DDQN(alpha=0.0003, state_dim=env.observation_space.shape[0], action_dim=env.action_space.n,
                 fc1_dim=256, fc2_dim=256, ckpt_dir=args.ckpt_dir, gamma=0.99, tau=0.005, epsilon=1.0,
                 eps_end=0.05, eps_dec=5e-4, max_size=1000000, batch_size=256)
    # create_directory(args.ckpt_dir, sub_dirs=['Q_eval', 'Q_target'])
    total_rewards, avg_rewards, eps_history = [], [], []

    for episode in range(args.max_episodes):
        total_reward = 0
        done = False
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation, isTrain=True)
            observation_, reward, done, info = env.step(action)
            agent.remember(observation, action, reward, observation_, done)
            agent.learn()
            total_reward += reward
            observation = observation_

        total_rewards.append(total_reward)
        avg_reward = np.mean(total_rewards[-100:])
        avg_rewards.append(avg_reward)
        eps_history.append(agent.epsilon)
        print('EP:{} reward:{} avg_reward:{} epsilon:{}'.
              format(episode + 1, total_reward, avg_reward, agent.epsilon))

        # if (episode + 1) % 50 == 0:
        #     agent.save_models(episode + 1)

    episodes = [i for i in range(args.max_episodes)]
    plot_learning_curve(episodes, avg_rewards, 'Reward', 'reward', args.reward_path)
    plot_learning_curve(episodes, eps_history, 'Epsilon', 'epsilon', args.epsilon_path)


if __name__ == '__main__':
    main()