import torch
import torch.nn as nn
#Kfold
from sklearn.model_selection import KFold
from example import generate_data
import numpy as np


#implement ridge regression in torch
class Ridge_nonlinear(nn.Module):
    def __init__(self, input_size, output_size,f=nn.Identity()):
        super(Ridge_nonlinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.f = f

    def forward(self, x):
        #X batch,features
        out = self.linear(x)
        out = self.f(out)
        return out
    def fit(self,X,y,alpha=0.1,lr=0.00001,epochs=1000):
        #X batch,features
        #y batch,1

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(X)
            loss = torch.sum((outputs-y)**2)+alpha*torch.sum(self.linear.weight**2)
            loss.backward()
            
            optimizer.step()
            
    def predict(self,X):
        with torch.no_grad():
            outputs = self.forward(X)
        return outputs
    
    def score(self,X,y):
        with torch.no_grad():
            outputs = self.predict(X)
            u = torch.sum((outputs-y)**2)
            v = torch.sum((y-torch.mean(y))**2)
            score = 1-u/v
        return score
    
    def tune(self,X,y,kfold=5,alpha_range=[-5,5],lr=0.00001,epochs=1000):
        inner_fold = KFold(n_splits=kfold, shuffle=False)
        outer_fold = KFold(n_splits=kfold, shuffle=False)
        alpha_range = np.logspace(alpha_range[0],alpha_range[1],kfold)
        result_dict = {'best_alphas':[],'best_scores':[],'y_pred':[],'coefs':[]}
        best_score = 0
        for train_val_index,test_index in outer_fold.split(X):
            for iter,(train_index,val_index) in enumerate(inner_fold.split(X[train_val_index])):
                X_train,X_val = X[train_index],X[val_index]
                y_train,y_val = y[train_index],y[val_index]
                alpha = alpha_range[iter]
                self.fit(X_train,y_train,alpha=alpha,lr=lr,epochs=epochs)
                score = self.score(X_val,y_val)
                if score>best_score:
                    best_score = score
                    best_alpha = alpha
            result_dict['best_alphas'].append(best_alpha)
            self.fit(X[train_val_index],y[train_val_index],lr=lr,alpha=best_alpha,epochs=epochs)
            y_hat = self.predict(X[test_index])
            score = self.score(X[test_index],y[test_index])
            print('score:{}'.format(score))
            result_dict['best_scores'].append(score)
            result_dict['y_pred'].append(y_hat)
        return result_dict

if __name__ == '__main__':
    #generate data
    X,y = generate_data(10000,sigmoid=True)
    y = torch.as_tensor(y, dtype=torch.float32)
    X = torch.as_tensor(X, dtype=torch.float32)
    #tune
    model = Ridge_nonlinear(100,1,torch.sigmoid)
    result = model.tune(X,y,5)
    
    #show result
    print(result['best_alphas'])
    print(result['best_scores'])
    y_hat = torch.cat(result['y_pred'])
    u = torch.sum((y_hat-y)**2)
    v = torch.sum((y-torch.mean(y))**2)
    score = 1-u/v
    print(score)