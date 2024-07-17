import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
import numpy as np
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
 
class DAE(nn.Module):
    def __init__(self,dropout_prob=0.1):
        super(DAE,self).__init__()
        """Encode in:3,256,256:out:1,4,4"""
        self.conv1_en = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,stride=1,padding=1)
        self.conv2_en = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3,stride=1,padding=1)
        self.conv3_en = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3,stride=1,padding=1)
        self.conv4_en = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3,stride=1,padding=1)
        self.conv5_en = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3,stride=1,padding=1)
        self.conv6_en = nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                    
    def encode(self, input):
        hidden1 = F.leaky_relu(self.conv1_en(input))
        hidden2 = self.pool(hidden1)  # 32 128 128
        hidden3 = F.leaky_relu(self.conv2_en(hidden2)) 
        hidden3 = self.pool(hidden3)  # 16 64 64
        hidden4 = F.leaky_relu(self.conv3_en(hidden3)) 
        hidden4 = self.pool(hidden4)  # 8 32 32
        hidden5 = F.leaky_relu(self.conv4_en(hidden4))  
        hidden5 = self.pool(hidden5)  # 4 16 16
        hidden6 = F.leaky_relu(self.conv5_en(hidden5))  
        hidden6 = self.pool(hidden6)  # 4 8 8
        output = F.leaky_relu(self.conv6_en(hidden6))  
        output = self.pool(output)  # 1 4 4
        return output
                
    def forward(self, input):
        encode_output= self.encode(input)
        return encode_output   

# 高斯过程分类器
class GPC:
    def __init__(self):
        kernel = 10000* RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e9))
        self.gpc = GaussianProcessClassifier(kernel=kernel,
                                        n_restarts_optimizer=5,
                                        max_iter_predict=100,  
                                        random_state=42)  
    def fit(self, X, y):
        self.gpc.fit(X, y)
    
    def predict(self, X):
        return self.gpc.predict(X)
    
    def predict_proba(self, X):
        return self.gpc.predict_proba(X)
    
class KSVM:
    def __init__(self, kernel='rbf', C=17000, gamma=0.5, random_state=100):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.random_state = random_state
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state)
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
class LSVM:
    def __init__(self, C=1800.0, random_state=100):
        self.C = C
        self.random_state = random_state
        self.model = SVC(kernel='linear', C=C, random_state=random_state, probability=True)
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class RandomForest:
    def __init__(self, n_estimators=200, max_depth=None, random_state=200):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LogisticReg:
    def __init__(self, C=13000, max_iter=5000, random_state=200):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class LightGBMClassifier:
    def __init__(self, num_leaves=36, learning_rate=0.1, n_estimators=200, random_state=200):
        self.num_leaves = num_leaves
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = lgb.LGBMClassifier(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators, random_state=random_state)
        
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

class KNN:
    def __init__(self, k=1):
        self.k = k
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
            y_pred = []
            for x in X_test:
                distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
                nearest_indices = np.argsort(distances)[:self.k]
                nearest_labels=[]
                for index in nearest_indices:
                    nearest_labels.append(self.y_train[index])   
                pred_label = np.argmax(np.bincount(nearest_labels))
                y_pred.append(pred_label)
            return np.array(y_pred)