import numpy as np
import pandas as pd

def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing feature.
  - y: A numpy array of shape (num_samples, ) containing label
  Returns:
  - w: a numpy array of shape (D, )
  """
  X_transpose = np.transpose(X)
  XXtranspose_inv = np.linalg.inv(np.dot(X_transpose,X))
  w = np.dot(np.dot(XXtranspose_inv,X_transpose),y)
  return w

def regularized_linear_regression(X, y, lambd):
  """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing feature.
    - y: A numpy array of shape (num_samples, ) containing label
    - lambd: a float number containing regularization strength
    Returns:
    - w: a numpy array of shape (D, )
    """
  [N, D] = X.shape
  regularizer = lambd*np.identity(D)
  X_transpose = np.transpose(X)
  XXtranspose_inv = np.linalg.inv(np.dot(X_transpose,X) + regularizer)
  w = np.dot(np.dot(XXtranspose_inv,X_transpose),y)
  return w

def tune_lambda(Xtrain, ytrain, Xval, yval, lambds):
  """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training feature.
    - ytrain: A numpy array of shape (num_training_samples, ) containing training label
    - Xval: A numpy array of shape (num_val_samples, D) containing validation feature.
    - yval: A numpy array of shape (num_val_samples, ) containing validation label
    - lambds: a list of lambdas
    Returns:
    - bestlambda: the best lambda you find in lambds
    """
  lambda_performance = {}
  for l in lambds:
    w = regularized_linear_regression(Xtrain, ytrain, l)
    err = test_error(w, Xval, yval)
    lambda_performance[l] = err
  bestlambda = min(lambda_performance, key=lambda_performance.get)
  return bestlambda

def test_error(w, X, y):
  """
    Compute the mean squre error on test set given X, y, and model parameter w.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test feature.
    - y: A numpy array of shape (num_samples, ) containing test label
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
  [N, D] = X.shape
  w_tranpose = np.transpose(w)
  predictions = np.dot(X, w_tranpose)
  error_square = np.square(y-predictions)
  sum_error_square = np.sum(error_square)
  err = (sum_error_square/N)
  return err

def data_processing():
  white = pd.read_csv('winequality-white.csv', low_memory=False, sep=';').values

  [N, d] = white.shape

  np.random.seed(3)
  # prepare data
  ridx = np.random.permutation(N)
  ntr = int(np.round(N * 0.8))
  nval = int(np.round(N * 0.1))
  ntest = N - ntr - nval

  # spliting training, validation, and test

  Xtrain = np.hstack([np.ones([ntr, 1]), white[ridx[0:ntr], 0:-1]])

  ytrain = white[ridx[0:ntr], -1]

  Xval = np.hstack([np.ones([nval, 1]), white[ridx[ntr:ntr + nval], 0:-1]])
  yval = white[ridx[ntr:ntr + nval], -1]

  Xtest = np.hstack([np.ones([ntest, 1]), white[ridx[ntr + nval:], 0:-1]])
  ytest = white[ridx[ntr + nval:], -1]
  return Xtrain, ytrain, Xval, yval, Xtest, ytest


def main():
  np.set_printoptions(precision=3)
  Xtrain, ytrain, Xval, yval, Xtest, ytest = data_processing()
  w = linear_regression_noreg(Xtrain, ytrain)
  print("dimensionality of the model parameter is ", len(w), ".", sep="")
  print("model parameter is ", np.array_str(w))
  
  lambd = 5.0
  wl = regularized_linear_regression(Xtrain, ytrain, lambd)
  print("\n")
  print("dimensionality of the model parameter is ", len(wl), sep="")
  print("lambda = ", lambd, ", model parameter is ", np.array_str(wl), sep="")

  lambds = [0, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10, 10 ** 2]
  bestlambd = tune_lambda(Xtrain, ytrain, Xval, yval, lambds)
  print("\n")
  print("tuning lambda, the best lambda =  ", bestlambd, sep="")

  wbest = regularized_linear_regression(Xtrain, ytrain, bestlambd)
  mse = test_error(wbest, Xtest, ytest)
  print("\n")
  print("Mean Square Error(MSE) on test is %.3f" % mse)
  
if __name__ == "__main__":
    main()
    