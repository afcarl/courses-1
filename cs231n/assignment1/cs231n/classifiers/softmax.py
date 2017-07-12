import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  num_classes = W.shape[1]
  loss = 0.0
  dW = np.zeros_like(W)
  for i in range(num_train):
        unnormalized_probs = np.e**np.matmul(X[i],W)
        sum_unnormalized_probs = np.sum(unnormalized_probs)
        normalized_probs = unnormalized_probs/sum_unnormalized_probs
        loss+= -1*np.log(normalized_probs[y[i]])
        for j in range(num_classes):
            if j == y[i]:
                grad = (normalized_probs[y[i]]-1)*X[i]
                dW[:,j] += grad.T
            else:
                grad = normalized_probs[j]*X[i]
                dW[:,j] += grad.T
  loss += 0.5*reg*np.sum(W*W)
  loss /= num_train
  dW += reg*W
  dW /= num_train

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  unnormalized_probs = np.e**np.matmul(X,W)
  sum_unnormalized_probs = np.sum(unnormalized_probs, axis = 1).reshape(num_train,1)
  normalized_probs = unnormalized_probs/sum_unnormalized_probs
  loss = (np.sum(-1*np.log(np.choose(y, normalized_probs.T))) + reg*np.sum(W*W))/num_train
  grads = normalized_probs
  grads[np.arange(grads.shape[0]), y] -=1
  dW = (np.matmul(X.T,grads)+reg*W)/num_train

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

