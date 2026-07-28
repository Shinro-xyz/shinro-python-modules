## sliding mode controller, SMC implementation with state space form

from components import Controller

class SlidingModeController(Controller):
    def __init__(self, k1:float, k2:float, phi:float, alpha:float,