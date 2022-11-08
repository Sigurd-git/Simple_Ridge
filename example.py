import numpy as np
def genrate_data():
    # Generate data for the example of linear regression.
    # X is a 100x100 matrix, Y is a 100x1 vector.
    # W is a 100x1 vector, which is the weight coefficient of the regression.
    # b is a scalar, which is the bias of the regression.
    X = np.random.randn(100,100)
    W = np.random.randn(100,1)
    b = np.random.randn(1)
    Y = X @ W  + b 
    return X,Y,W,b
if __name__ == '__main__':
    X,Y,W,b = genrate_data()
    print('X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape))
    print('W.shape:{}, b.shape:{}'.format(W.shape,b.shape))