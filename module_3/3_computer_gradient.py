# UNQ_C3
# GRADED FUNCTION: compute_gradient
def compute_gradient(X, y, w, b, *argv): 
    """
    Computes the gradient for logistic regression 
 
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns
      dj_dw : (ndarray Shape (n,)) The gradient of the cost w.r.t. the parameters w. 
      dj_db : (scalar)             The gradient of the cost w.r.t. the parameter b. 
    """
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    ### START CODE HERE ### 
    for i in range(m):              # iterate over i training examples
        z_wb = np.dot(X[i], w)      # take dot product of ith training example and w
        z_wb += b                   # add bias to dot product
        f_wb = sigmoid(z_wb)        # Insert (dot product + bias) into sigmoid function
        difference = f_wb - y[i]    # take difference of sigmoid result and ith target
        dj_db += difference         # increment dj_db by difference

        for j in range(n):   # use for loop for dj_dw
            dj_dw[j] += (difference) * X[i][j] # increment jth value of dj_dw by product of difference
                                               # and the element of X in ith row and jth column
    dj_dw /= m
    dj_db /= m
    ### END CODE HERE ###

        
    return dj_db, dj_dw