import numpy as np
from task_inference_utils.base_inference import BaseTaskInference

class SimpleTaskInference(BaseTaskInference):
    def infer_task(self, trajectory_buffer):
        if len(trajectory_buffer) == 0:
            return np.zeros(self.context_size, dtype=np.float32)
        
        avg_reward = np.mean([transition[2] for transition in trajectory_buffer])
        return 0*np.array([avg_reward] * self.context_size, dtype=np.float32)