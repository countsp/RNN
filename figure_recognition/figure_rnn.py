# %%
# 导入必要的库
import numpy as np  # 用于处理数据的线性代数操作
import pandas as pd  # 用于数据处理，例如 CSV 文件的输入/输出
import matplotlib.pyplot as plt  # 用于绘图
import os  # 操作系统库，用于访问文件系统
print(os.listdir("./input"))  # 打印出 ./input 文件夹中的所有文件

# %%
# 导入 PyTorch 库和其他必要模块
import torch  # PyTorch 的主库
import torch.nn as nn  # 神经网络模块
from torch.autograd import Variable  # 用于构建可计算梯度的变量
from sklearn.model_selection import train_test_split  # 用于划分训练集和测试集
from torch.utils.data import DataLoader, TensorDataset  # 用于数据加载和处理

# %%
# 准备数据集
# 从 CSV 文件中加载训练数据，数据类型为 float32
train = pd.read_csv(r"./input/train.csv", dtype=np.float32)

# 将数据拆分为特征（像素值）和标签（0-9 的数字）
targets_numpy = train.label.values  # 标签为 0-9 的数字
features_numpy = train.loc[:, train.columns != "label"].values / 255  # 特征值归一化到 0-1 之间

# 将数据拆分为训练集和测试集，80% 为训练集，20% 为测试集
features_train, features_test, targets_train, targets_test = train_test_split(
    features_numpy, targets_numpy, test_size=0.2, random_state=42)

# 将训练集中的特征和标签转换为 PyTorch 的 Tensor
featuresTrain = torch.from_numpy(features_train)  # 特征数据
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor)  # 标签数据，数据类型为 long

# 将测试集中的特征和标签转换为 PyTorch 的 Tensor
featuresTest = torch.from_numpy(features_test)  # 特征数据
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor)  # 标签数据，数据类型为 long

# 设置 batch_size，epoch，和迭代次数
batch_size = 100  # 每次训练时处理的样本数量
n_iters = 10000  # 总的迭代次数
num_epochs = n_iters / (len(features_train) / batch_size)  # 计算 epoch 次数
num_epochs = int(num_epochs)

# 创建 PyTorch 数据集和数据加载器
train = TensorDataset(featuresTrain, targetsTrain)  # 训练集
test = TensorDataset(featuresTest, targetsTest)  # 测试集

# 使用 DataLoader 将数据集加载到训练和测试加载器中
train_loader = DataLoader(train, batch_size=batch_size, shuffle=False)  # 训练数据加载器
test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)  # 测试数据加载器

# 可视化数据集中一张图片
plt.imshow(features_numpy[10].reshape(28, 28))  # 将第 10 张图片重塑为 28x28 的图像
plt.axis("off")  # 去除坐标轴
plt.title(str(targets_numpy[10]))  # 显示对应的标签
plt.savefig('graph.png')  # 保存图片
plt.show()  # 显示图片

# %%
# 创建 RNN 模型
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()
        
        # 定义隐藏层的维度
        self.hidden_dim = hidden_dim
        
        # 定义隐藏层的数量
        self.layer_dim = layer_dim
        
        # 定义 RNN 层
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        
        # 定义全连接层，用于输出
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        
        # 初始化隐藏状态为零
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        
        # RNN 前向传播（单步）
        out, hn = self.rnn(x, h0)
        
        # 取最后一个时间步的输出，并通过全连接层
        out = self.fc(out[:, -1, :]) 
        return out

# 设置模型的超参数
input_dim = 28  # 输入的维度（每次输入一行的 28 个像素）
hidden_dim = 100  # 隐藏层维度
layer_dim = 1  # 隐藏层数量
output_dim = 10  # 输出维度（0-9 的数字分类）

# 创建 RNN 模型
model = RNNModel(input_dim, hidden_dim, layer_dim, output_dim)

# 定义损失函数，使用交叉熵损失
error = nn.CrossEntropyLoss()

# 使用 SGD 优化器，设置学习率
learning_rate = 0.05
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# %%
seq_dim = 28  # RNN 每次输入 28 个时间步（对应图片的每一行）

# 初始化一些变量用于存储损失和准确率
loss_list = []  # 存储每次迭代的损失
iteration_list = []  # 存储迭代次数
accuracy_list = []  # 存储每次迭代的准确率
count = 0  # 用于记录迭代次数

# 训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        # 将图片数据转换为 RNN 的输入格式 (batch_size, seq_dim, input_dim)
        train = Variable(images.view(-1, seq_dim, input_dim))
        labels = Variable(labels)
        
        # 清空上一步的梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(train)
        
        # 计算损失
        loss = error(outputs, labels)
        
        # 反向传播计算梯度
        loss.backward()
        
        # 更新模型参数
        optimizer.step()
        
        count += 1
        
        # 每 250 次迭代计算一次准确率
        if count % 250 == 0:
            correct = 0
            total = 0
            
            # 在测试集上计算模型的准确率
            for images, labels in test_loader:
                images = Variable(images.view(-1, seq_dim, input_dim))
                
                # 前向传播，得到模型预测
                outputs = model(images)
                
                # 取出预测结果中每行最大值对应的索引，即预测的分类
                predicted = torch.max(outputs.data, 1)[1]
                
                # 总的样本数量
                total += labels.size(0)
                
                # 计算预测正确的数量
                correct += (predicted == labels).sum()
            
            # 计算准确率
            accuracy = 100 * correct / float(total)
            
            # 记录损失和准确率
            loss_list.append(loss.item())
            iteration_list.append(count)
            accuracy_list.append(accuracy)
            
            # 每 500 次迭代输出一次损失和准确率
            if count % 500 == 0:
                print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.item(), accuracy))


# %%
# 可视化损失曲线
plt.plot(iteration_list, loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("RNN: Loss vs Number of iteration")
plt.show()

# %%
# 可视化准确率曲线
plt.plot(iteration_list, accuracy_list, color="red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("RNN: Accuracy vs Number of iteration")
plt.savefig('graph.png')
plt.show()



