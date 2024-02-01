# time: 2023/10/21 21:17
# author: YanJP
import gym
from model import Actor_Critic
import matplotlib.pyplot as plt


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    model = Actor_Critic(env)  #实例化Actor_Critic算法类
    reward = []
    for episode in range(200):
        s = env.reset()  #获取环境状态
        env.render()     #界面可视化
        done = False     #记录当前回合游戏是否结束
        ep_r = 0
        while not done:
            # 通过Actor_Critic算法对当前环境做出行动
            a,log_prob = model.get_action(s)

            # 获得在做出a行动后的最新环境
            s_,rew,done,_  = env.step(a)

            #计算当前reward
            ep_r += rew

            #训练模型
            model.learn(log_prob,s,s_,rew)

            #更新环境
            s = s_
        reward.append(ep_r)
        print(f"episode:{episode} ep_r:{ep_r}")
    plt.plot(reward)
    plt.show()
