from abc import ABC, abstractmethod

class BaseTaskInference(ABC):
    def __init__(self, context_size):
        self.context_size = context_size

    @abstractmethod
    def infer_task(self, trajectory_buffer):
        """
        Abstract method to infer the current task based on the trajectory buffer.
        """
        pass