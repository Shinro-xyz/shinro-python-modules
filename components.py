from abc import ABC, abstractmethod
from typing import Any

class Controller(ABC):
    @abstractmethod
    def compute(self,*args:Any,**kwargs:Any)-> Any:
        pass

    def reset(self,*args:Any,**kwargs:Any)-> Any:
        pass

class Plant(ABC):
    @abstractmethod
    def get_state(self):
        pass

    @abstractmethod
    def get_model(self):
        pass

    @abstractmethod
    def step(self, *args, **kwargs):
        pass