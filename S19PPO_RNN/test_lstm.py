# # time: 2024/1/22 23:15
# # author: YanJP
# import torch
# import torch.nn as nn
#
# input_size = 5
# sequence_length = 10
# batch_size = 10  # 将 batch_size 修改为匹配期望大小的值
#
# input_sequence = torch.randn(sequence_length, batch_size, input_size)
#
# hidden_size = 8
# num_layers = 2
# bidirectional = False
#
# lstm_layer = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
#
# # 修改 batch_size 以匹配期望的隐藏状态大小
# h0 = torch.zeros(num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
# c0 = torch.zeros(num_layers * (2 if bidirectional else 1), batch_size, hidden_size)
#
# output, (hn, cn) = lstm_layer(input_sequence, (h0, c0))
#
# print("Output shape:", output.shape)
# print("Final hidden state shape:", hn.shape)
# print("Final cell state shape:", cn.shape)

import torch
import torch.nn as nn
import torch.optim as optim

# 假设有训练数据 train_data 和对应的标签 target_data

input_size = 5
hidden_size = 8
num_layers = 2
bidirectional = False
batch_size = 10
sequence_length = 10

# 创建模型
lstm_model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# 模型训练
for epoch in range(10000):  # 假设进行100个训练周期
    # 假设 train_data 和 target_data 是你的训练数据和标签
    input_sequence = torch.randn(batch_size, sequence_length, input_size)
    target_sequence = torch.randn(batch_size, sequence_length, hidden_size)

    # 将梯度置零
    optimizer.zero_grad()

    # 前向传播
    output, _ = lstm_model(input_sequence)

    # 计算损失
    loss = criterion(output, target_sequence)

    # 反向传播
    loss.backward()

    # 参数更新
    optimizer.step()

    # 输出当前训练周期的损失
    print(f'Epoch {epoch+1}/{100}, Loss: {loss.item()}')
