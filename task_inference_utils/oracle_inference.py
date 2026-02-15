import numpy as np
from task_inference_utils.base_inference import BaseTaskInference

class OracleInference(BaseTaskInference):
    def __init__(self, task_size):
        self.task_size = task_size
        self.context_size = task_size


    def infer_task(self, cur_task):
        
        condition = np.zeros(self.task_size)
        condition[cur_task] = 1
        return condition