
# UNQ_C2
# GRADED FUNCTION: compute_cost
def compute_cost(X, y, w, b, *argv):
    """
    Computes the cost over all examples
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      y : (ndarray Shape (m,))  target value 
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model
      *argv : unused, for compatibility with regularized version below
    Returns:
      total_cost : (scalar) cost 
    """

    m, n = X.shape

    ### START CODE HERE ###
    
    total_cost = 0
    z = np.dot(X, w) + b
    for i in range(m):
        s = sigmoid(z[i])
        if y[i] == 1:
            total_cost += -y[i] * np.log(s) 
        elif y[i] == 0:
             total_cost += (-1 - y[i]) * np.log(1 - s)
                            
    total_cost /= m
        
    ### END CODE HERE ### 

    return total_cost
