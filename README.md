# RNN 
RNN用于处理时间序列数据：在MLP的基础上，第一次处理的输出给了第二次处理

![Screenshot from 2024-10-14 13-22-21](https://github.com/user-attachments/assets/0bdaefe2-bc4c-4dcc-af06-f648e684a43e)

**缺陷：** 在向后传播时，前面的信息逐渐被忽视，距离越远忘记越多。

![Screenshot from 2024-10-14 13-32-24](https://github.com/user-attachments/assets/087b88a0-3c37-4723-aa06-fad7523bd205)

# LSTM

增加记忆细胞，传递远处的重要信息。
![Screenshot from 2024-10-14 13-45-44](https://github.com/user-attachments/assets/0713c963-44d3-49ae-9afd-813914c5df79)

增加不同的门，实现记忆遗忘，解决了RNN的梯度消失的问题。

![Screenshot from 2024-10-14 13-55-30](https://github.com/user-attachments/assets/051439bc-f165-4087-a4b0-435c47cac143)
