# UNQ_C4
# GRADED FUNCTION: predict

def predict(X, w, b): 
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w
    
    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model      
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape   
    p = np.zeros(m)
   
    ### START CODE HERE ### 
    for i in range(m):           # Loop over each example  
        z_wb = np.dot(X[i], w)   # take dot product of ith example and w
        z_wb += b                # Add bias to dot product
        f_wb = sigmoid(z_wb)     # Insert (dot product + bias) into sigmoid function
        if f_wb >= 0.5:          # if f_wb >= 0.5
            p[i] = 1
        else:
            p[i] = 0             # else if f_wb < 0.5
        
    ### END CODE HERE ### 
    return p