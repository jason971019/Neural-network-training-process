
# coding: utf-8

# ## Import the library

# In[2]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
np.set_printoptions(threshold=np.inf)

# ## layer definition (Need to do!!!)

# In[3]:


def InnerProduct_For(x,W,b):
    y = np.dot(x,W)+b
    return y

def InnerProduct_Back(dEdy,x,W,b):
    dEdx = np.dot(dEdy,W.T)
    dEdW = np.dot(dEdy.T,x)
    dEdb = np.dot(np.ones([1,48000]),dEdy)
    return dEdx,dEdW,dEdb

def Softmax_For(x):
    softmax = (np.exp(x).T/np.sum(np.exp(x),axis=1).T).T
    return softmax

def Softmax_Back(y,t):
    dEdx = y-t
    return dEdx

def Sigmoid_For(x):
    y = 1/(1+np.exp(-x))
    return y

def Sigmoid_Back(dEdy,x):
    dEdx = np.dot(dEdy.T,np.exp(-x)/pow(1+np.exp(-x),2))
    return dEdx

def ReLu_For(x):
    y = np.maximum(x,0)
    return y

def ReLu_Back(dEdy,x):
    x = np.int64(x>0)
    dEdx = dEdy*x
    return dEdx

def loss_For(y,y_pred):
    loss = np.square(pow(y-y_pred,2))
    return loss


# ## Setup the Parameters and Variables (Can tune that!!!)

# In[4]:

eta = 0.000001       #learning rate
Data_num = 784      #size of input data   (inputlayer)
W1_num = 15         #size of first neural (1st hidden layer)
Out_num = 10        #size of output data  (output layer)
iteration = 3000    #epoch for training   (iteration)
image_num = 60000   #input images
test_num  = 10000   #testing images

## Cross Validation ##
##spilt the training data to 80% train and 20% valid##
train_num = int(image_num*0.8)
valid_num = int(image_num*0.2)

# ## Setup the Data (Create weight array here!!!)

# In[5]:

w_1= (np.random.normal(0,1,Data_num*W1_num)).reshape(Data_num,W1_num)/100
w_out  = (np.random.normal(0,1,W1_num*Out_num)).reshape(W1_num, Out_num)/100
b_1, b_out = randn(1,W1_num)/100,randn(1,Out_num)/100
"""
print("w1 shape:", w_1.shape) #(784, 15)
print("w_out shape:", w_out.shape) #(15, 10)
print("b_1 shape:", b_1.shape) #(1, 15)
print("b_out shape:", b_out.shape) #(1, 10)
print(w_1)
"""

# ## Prepare all the data

# ### Load the training data and labels from files

# In[6]:

df = pd.read_csv('fashion-mnist_train_data.csv')
fmnist_train_images = df.as_matrix()
"""
print("Training data:",fmnist_train_images.shape[0]) #60000
print("Training data shape:",fmnist_train_images.shape) #(60000, 784)
"""

df = pd.read_csv('fashion-mnist_test_data.csv')
fmnist_test_images = df.as_matrix()
"""
print("Testing data:",fmnist_test_images.shape[0]) #10000
print("Testing data shape:",fmnist_test_images.shape) #(10000, 784)
"""

df = pd.read_csv('fashion-mnist_train_label.csv')
fmnist_train_label = df.as_matrix()
"""
print("Training labels shape:",fmnist_train_label.shape) #(60000, 1)
"""

# ### Show the 100 testing images

# In[7]:

"""
plt.figure(figsize=(20,20))
for index in range(100):
    image = fmnist_test_images[index].reshape(28,28)
    plt.subplot(10,10,index+1,)
    plt.imshow(image)
plt.show() 
"""

# ### Convert the training labels data type to one hot type

# In[8]:

label_temp = np.zeros((image_num,10), dtype = np.float32)
for i in range(image_num):
    label_temp[i][fmnist_train_label[i][0]] = 1
train_labels_onehot = np.copy(label_temp)
"""
print("Training labels shape:",train_labels_onehot.shape) #(60000, 1)
"""

# ### Separate train_images, train_labels into training and validating 

# In[13]:

train_data_img = np.copy(fmnist_train_images[:train_num,:])
train_data_lab = np.copy(train_labels_onehot[:train_num,:])
valid_data_img = np.copy(fmnist_train_images[train_num:,:])
valid_data_lab = np.copy(train_labels_onehot[train_num:,:])

# Normalize the input data between (0,1)
train_data_img = train_data_img/255.
valid_data_img = valid_data_img/255.
test_data_img = fmnist_test_images/255.
"""
print("Train images shape:",train_data_img.shape) #(48000, 784)
print("Train labels shape:",train_data_lab.shape) #(48000, 10)
print("Valid images shape:",valid_data_img.shape) #(12000, 784)
print("Valid labels shape:",valid_data_lab.shape) #(12000, 10)
print("Test  images shape:",test_data_img.shape) #(10000, 784)
print(train_data_lab)
"""

# ## Execute the Iteration (Need to do!!!)

# In[12]:

valid_accuracy = []

y = np.zeros([15,1])

for i in range(iteration):
    # Forward-propagation
    y_1 = InnerProduct_For(train_data_img,w_1,b_1)
    y_2 = ReLu_For(y_1)
    y_3 = InnerProduct_For(y_2,w_out,b_out)
    y_4 = Softmax_For(y_3)
    loss = loss_For(train_data_lab,y_4)
    
    # Bakcward-propagation
    dEdx = Softmax_Back(y_4,train_data_lab)
    dE1dx1, dE1dW1, dE1db1 = InnerProduct_Back(dEdx, y_2, w_out, b_out)
    dE2dx2 = ReLu_Back(dE1dx1, y_2)
    dE3dx3, dE3dW3, dE3db3 = InnerProduct_Back(dE2dx2, train_data_img, w_1, b_1)
    
    # Parameters Updating (Gradient descent)
    w_1 = w_1-eta*dE3dW3.T
    b_1 = b_1-eta*dE3db3
    w_out = w_out-eta*dE1dW1.T
    b_out = b_out-eta*dE1db1
    
    # Do cross-validation to evaluate model
    y_valid1 = InnerProduct_For(valid_data_img,w_1,b_1)
    y_valid2 = ReLu_For(y_valid1)
    y_valid3 = InnerProduct_For(y_valid2,w_out,b_out)
    y_valid4 = Softmax_For(y_valid3)
    
    # Get 1-D Prediction array
    valid_pre = np.argmax(y_valid4, axis=1)
    valid_lab = np.argmax(valid_data_lab, axis=1)

    # Compare the Prediction and validation
    true = 0
    for j in range(y_valid4.shape[0]):
        if valid_lab[j] == valid_pre[j]:
            true += 1
    
    #Calculate the accuracy
    accuracy = true/12000*100
    valid_accuracy.append(accuracy)
    
# ## Testing Stage

# ### Predict the test images (Do forward propagation again!!!)

# In[10]:

# Forward-propagation
    y_test1 = InnerProduct_For(test_data_img,w_1,b_1)
    y_test2 = ReLu_For(y_test1)
    y_test3 = InnerProduct_For(y_test2,w_out,b_out)
    test_Out_data = Softmax_For(y_test3)

# ### Convert results to csv file (Input the (10000,10) result array!!!)

# In[12]:

# Convert "test_Out_data" (shape: 10000,10) to "test_Prediction" (shape: 10000,1)
test_Prediction = np.argmax(test_Out_data, axis=1)[:,np.newaxis].reshape(test_num,1)
df = pd.DataFrame(test_Prediction,columns=["Prediction"])
df.to_csv("DL_LAB1_prediction_ID.csv",index=True, index_label="index")

# ## Convert results to csv file

# In[16]:

accuracy = np.array(valid_accuracy)
plt.plot(accuracy, label="$iter-accuracy$")
y_ticks = np.linspace(0, 100, 11)
plt.legend(loc='best')
plt.xlabel('iteration')
plt.axis([0, iteration, 0, 100])
plt.ylabel('accuracy')
plt.show()
