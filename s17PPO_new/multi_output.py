# time: 2023/11/21 21:33
# author: YanJP
#这是对应多维连续变量的例子：
# 参考链接：https://github.com/XinJingHao/PPO-Continuous-Pytorch/blob/main/utils.py
# https://www.zhihu.com/question/417161289
import torch.nn as nn
import torch
class Policy(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, num_outputs):
        super(Policy, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.ReLU(True),
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.ReLU(True),
            nn.Linear(n_hidden_2, num_outputs)
        )

class Normal(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.stds = nn.Parameter(torch.zeros(num_outputs))
    def forward(self, x):
        dist = torch.distributions.Normal(loc=x, scale=self.stds.exp())
        action = dist.sample()
        return action

if __name__ == '__main__':
    policy = Policy(4,20,20,5)
    normal = Normal(5)
    observation = torch.Tensor(4)
    action = normal.forward(policy.layer( observation))
    print("action: ",action)