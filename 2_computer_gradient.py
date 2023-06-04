# UNQ_C2
# GRADED FUNCTION: compute_gradient
def compute_gradient(x, y, w, b): 
    """
    Computes the gradient for linear regression 
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model  
    Returns
        dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
        dj_db (scalar): The gradient of the cost w.r.t. the parameter b     
        """

    # Number of training examples
    m = x.shape[0]

    # You need to return the following variables correctly
    dj_dw = 0
    dj_db = 0

    ### START CODE HERE ###

    # The prediction of the model for that example
    for i in range(m):
        prediction_i = w * x[i] + b
        dj_dw_i = (prediction_i - y[i]) * x[i]
        dj_db_i = prediction_i - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i

    dj_dw *= 1/m
    dj_db *= 1/m
    ### END CODE HERE ### 

    return dj_dw, dj_db