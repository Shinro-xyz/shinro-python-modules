from shinro.components import Controller
import numpy as np

# Model Predictive Path Integral (MPPI) controller
# optimal control algorithm, sampling based, information theoretic

class MPPI_Controller(Controller):
    def __init__(self):
        
