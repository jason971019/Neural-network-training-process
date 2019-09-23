# Neural-network-training-process
Using backpropagation and gradient descent to train a neural networks. Build a neural networks model for the classification problem.
### Forward-Propagation
- Inner-Product (Hidden Layer)：每一個輸出是前一層的每個結點乘以一個權重係數 (W)，再加上一個偏置值 (b)。
- Activation Function (Hidden Layer)：利用非線性函數作為 Activation function，使輸入不再是線性組合，得以逼近任意函數。
- Inner-Product (Output Layer)：每一個輸出是前一層的每個結點乘以一個權重係數 (W)，再加上一個偏置值 (b)。
- Softmax (Output Layer)：對向量進行歸一化，凸顯其中最大的值並抑制遠低於最大值的其他分量，藉此找出最可能的值。
### Backward-Propagation
- Softmax (Output Layer)：將預測結果與 Label 比較得到 Loss。
- Inner-Product (Output Layer)：由已知傳遞到該層的梯度 (dEdy)，透過 chain rule 運算得到 dEdx、dEdW、dEdb
- Activation Function (Hidden Layer)：保留已知傳遞到該層的梯度 (dEdy) 中大於 0 的值，小於 0 的值則指定為 0。
- Inner-Product (Hidden Layer)：由已知傳遞到該層的梯度 (dEdy)，透過 chain rule 運算得到 dEdx、dEdW、dEdb
#### Layer functions
![image](https://github.com/jason971019/Neural-network-training-process/blob/master/Layer%20function.png)
#### Gradient descent
![image](https://github.com/jason971019/Neural-network-training-process/blob/master/Gradient%20descent.png)
#### Result
- iteration = 500
![image](https://github.com/jason971019/Neural-network-training-process/blob/master/500.png)
- iteration = 1000
![image](https://github.com/jason971019/Neural-network-training-process/blob/master/1000.png)
