import numpy as np
def generate_data(n,sigmoid=False):
    # Generate data for the example of linear regression.
    # X is a 100x100 matrix, Y is a 100x1 vector.
    # W is a 100x1 vector, which is the weight coefficient of the regression.
    # b is a scalar, which is the bias of the regression.
    X = np.random.randn(n,100)
    W = np.random.randn(100,1)
    b = np.random.randn(1)
    # error = np.random.randn(n,1)/100000
    Y = X @ W  + b
    if sigmoid:
        Y = 1/(1+np.exp(-Y))
    # Y = Y + error
    return X,Y
if __name__ == '__main__':
    X,Y = generate_data(10,sigmoid=True)
    print('X.shape:{}, Y.shape:{}'.format(X.shape,Y.shape))
    print('W.shape:{}, b.shape:{}'.format(W.shape,b.shape))