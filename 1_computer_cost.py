# UNQ_C1
# GRADED FUNCTION: compute_cost

def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    
    # You need to return this variable correctly
    cost_sum = 0
    
    ### START CODE HERE ###
    
    for i in range(m):
        prediction_i = w * x[i] + b 
        cost_i = (prediction_i - y[i]) ** 2
        cost_sum += cost_i
    
    ### END CODE HERE ### 
    total_cost = (1 / (2 * m)) * cost_sum
    return total_cost