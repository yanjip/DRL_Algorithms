# time: 2023/11/22 21:04
# author: YanJP


import torch
import torch
import torch.nn as nn
from torch.distributions import Categorical

class MultiDimensionalActor(nn.Module):
    def __init__(self, input_dim, output_dims):
        super(MultiDimensionalActor, self).__init__()

        # Define a shared feature extraction network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Define individual output layers for each action dimension
        self.output_layers = nn.ModuleList([
            nn.Linear(64, num_actions) for num_actions in output_dims
        ])

    def forward(self, state):
        # Feature extraction
        features = self.feature_extractor(state)

        # Generate Categorical objects for each action dimension
        categorical_objects = [Categorical(logits=output_layer(features)) for output_layer in self.output_layers]

        return categorical_objects

# 定义主函数
def main():
    # 定义输入状态维度和每个动作维度的动作数
    input_dim = 10
    output_dims = [5, 8]  # 两个动作维度，分别有 3 和 4 个可能的动作

    # 创建 MultiDimensionalActor 实例
    actor_network = MultiDimensionalActor(input_dim, output_dims)

    # 生成输入状态（这里使用随机数据作为示例）
    state = torch.randn(1, input_dim)

    # 调用 actor 网络
    categorical_objects = actor_network(state)

    # 输出每个动作维度的采样动作和对应的对数概率
    for i, categorical in enumerate(categorical_objects):
        sampled_action = categorical.sample()
        log_prob = categorical.log_prob(sampled_action)
        print(f"Sampled action for dimension {i+1}: {sampled_action.item()}, Log probability: {log_prob.item()}")

if __name__ == "__main__":
    main()
