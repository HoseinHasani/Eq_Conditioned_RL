from gymnasium import Wrapper
from collections import deque
import numpy as np
import ctypes


class NonStationaryEnv(Wrapper):
    """
    Implements a non-stationary environment wrapper for various OpenAI Gym environments.
    
    This wrapper allows the environment to change its dynamics, such as gravity, mass, friction, etc., at the start of each episode. The changes are controlled by a queue of task IDs, which are cycled through with each new episode.
    
    The wrapper supports the following environments:
    - CartPole-v1: Changes gravity and mass
    - HalfCheetah-v4: Changes gravity and wind force
    - Pendulum-v1: Changes gravity and pendulum length
    - Swimmer-v4: Changes viscosity, mass, friction, and degrees of freedom
    - Reacher-v4: Changes mass and degrees of freedom
    
    The wrapper keeps track of the true task labels for each episode, which can be used for supervised learning.
    """

    def __init__(self, env, max_ep_len, n_tasks, task_name, env_name, n_supervised_episodes=1):
        super(NonStationaryEnv, self).__init__(env)
        self.tasks = deque([i for i in range(n_tasks)])
        self.max_episode_len = max_ep_len
        self.task_name = task_name
        self.env_name = env_name
        self.n_supervised_episodes = n_supervised_episodes
        self.counter = 0
   
        self.h_winds = {0: np.array([0,0,0,0,0,0]), 1: np.array([10,0,0,0,0,0]),
         2: np.array([-10,0,0,0,0,0])}  
   
        self.slopes = {0: [0.5, 0, -0.866, 0],1: [0.9239, 0, -0.3827, 0], 2: [0.9659, 0, -0.2588, 0], 3: [1.0, 0, 0, 0], 4: [0.9659, 0, 0.2588, 0], 5: [0.9239, 0, 0.3827, 0], 6: [0.5, 0, 0.866, 0]}

        self.h_gravities =  {0 : -0.1, 1 : -9.81 , 2: -18.81, 3:-32.81 , 4: -55.81} #Gravities that are suitable for half cheetah
        
        self.c_gravities =  {0 : +9.81, 1 : -9.81 , 2: -27.81, 3:-47.81 , 4: -67.81} #Gravities that are suitable for Cartpole  
        
        self.p_gravities =  {0 : 3.81, 1 : 6.81 , 2: 9.81, 3: 12.81 , 4: 15.81} #Gravities for pendulum
        self.powers = {0 : 0.0015, 1 : 0.0045, 2 : 0.0075, 3 : 0.0105, 4 : 0.0135} #Powers for mountain car
        self.lengths =  {0 : 0.6, 1 : 0.8 , 2: 1, 3: 1.3 , 4: 1.7} #Lenghths for cartpole
        self.swimmer_masses = {0: np.array([  0. , 13.60471674, 13.60471674, 13.60471674]),
        1: np.array([ 0. , 14.60471674, 14.60471674, 14.60471674]),
        2: np.array([ 0. , 15.60471674, 15.60471674, 15.60471674]),
        3: np.array([ 0. , 16.60471674, 16.60471674, 16.60471674])}

        self.cartpole_masses = {0: np.array([ 0 , 1, 0.1]),
        1: np.array([ 0.2 , 1.2, 0.3]),
        2: np.array([ 0.4 , 1.4, 0.5]),
        3: np.array([ 0.6 , 1.6, 0.7])}

        self.reacher_masses = {0: np.array([ 0., 0.03560472, 0.03560472, 0.00418879, 0.00305363]),
        1: np.array([ 0., 0.23560472, 0.23560472, 0.02418879, 0.12305363]),
        2: np.array([ 0., 0.43560472, 0.43560472, 0.04418879, 0.24305363]),
        3: np.array([ 0., 0.63560472, 0.63560472, 0.06418879, 0.36305363])}


        self.viscosities =  {0 : 0, 1 : 50, 2: 100, 3: 150}

        self.dof =  {0 : np.array([0, 0, 0, 0, 0]), 1 : np.array([5, 5, 5, 5, 5]),
         2: np.array([10, 10, 10, 10, 10]), 3:np.array([15, 15, 15, 15, 15])}

        self.reacher_dof =  {0 : np.array([1, 1, 0, 0]), 1 : np.array([7, 7, 0, 0]),
         2: np.array([13, 13, 0, 0]), 3:np.array([19, 19, 0, 0])}
        
        self.geom_coef =  {0 : np.array([[1.e+00, 5.e-03, 1.e-04],[1.e+00, 5.e-03, 1.e-04],[1.e+00, 5.e-03, 1.e-04],[1.e+00, 5.e-03, 1.e-04]]),
         1 : np.array([[3.e+00, 25.e-03, 3.e-02],[3.e+00, 25.e-03, 3.e-02],[3.e+00, 25.e-03, 3.e-02],[3.e+00, 25.e-03, 3.e-02]]),
         2: np.array([[5.e+00, 45.e-03, 5.e-04],[5.e+00, 45.e-03, 5.e-04],[5.e+00, 45.e-03, 5.e-04],[5.e+00, 45.e-03, 5.e-04]]),
         3: np.array([[7.e+00, 65.e-03, 7.e-04],[7.e+00, 65.e-03, 7.e-04],[7.e+00, 65.e-03, 7.e-04],[7.e+00, 65.e-03, 7.e-04]])}

        self.lunar_winds = {0: -12.0, 1: 12.0}
        
        self.true_labels = []
        self.predicted_labels = []

    @property
    def current_task(self):
        return self.tasks[0]

    def step(self, action):
        self.counter += 1
        
        if self.env_name == "cartpole":
            if self.task_name == "gravity" or self.task_name == None:
                # Gravity Task for cartpole
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.env._env.physics.model.opt.gravity[:] = (ctypes.c_double * 3)(*[0., 0., self.c_gravities[self.current_task]])
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)
            if self.task_name == "mass":
                # Mass Task for cartpole
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.env._env.physics.model.body_mass =  self.cartpole_masses[self.current_task]
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)            
            else:
                assert False

       
        if self.env_name == "HalfCheetah-v4":
            if self.task_name == "gravity" or self.task_name == None:
                # Gravity Task for half Cheetah
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.unwrapped.model.opt.gravity[:] = (ctypes.c_double * 3)(*[0., 0., self.h_gravities[self.current_task]])
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)
            elif self.task_name == "wind":
                # Gravity Task for half Cheetah
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    for body_id in range(8):
                        self.unwrapped.data.xfrc_applied[body_id, :] = self.h_winds[self.current_task]
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)
            else:
                assert False

        elif self.env_name == "Pendulum-v1":
            if self.task_name == "gravity" or self.task_name == None:
                # Gravity Task for Penulum gym
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.unwrapped.g = self.p_gravities[self.current_task]
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)
            if self.task_name == "length":
                # Length Task for Penulum gym
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.unwrapped.l = self.lengths[self.current_task]
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)
            else:
                assert False
        
        elif self.env_name == "Swimmer-v4":
            if self.task_name == "viscosity" or self.task_name == None:
                # Viscosity Task swimmer
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.unwrapped.model.opt.viscosity = self.viscosities[self.current_task]
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)
            elif self.task_name == "mass": 
                # Mass Task for Swimmer
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.unwrapped.model.body_mass =  self.swimmer_masses[self.current_task]
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)
            elif self.task_name == "friction":
                # Friction coefficient Task swimmer
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.unwrapped.model.geom_friction = self.geom_coef[self.current_task]
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)
            elif self.task_name == "dof":
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.unwrapped.model.dof_damping = self.dof[self.current_task]
                    # print("v:",self.unwrapped.model.opt.viscosity)
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)
            else:
                assert False
                
        elif self.env_name == "Reacher-v4":
            if self.task_name == "mass" or self.task_name == None: 
                # Mass Task for Reacher
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.unwrapped.model.body_mass =  self.reacher_masses[self.current_task]
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)

            elif self.task_name == "dof":
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    self.unwrapped.model.dof_damping = self.reacher_dof[self.current_task]
                    # print("v:",self.unwrapped.model.opt.viscosity)
                    # print("SET TO TASK {} AT STEP {}!".format(self.current_task, self.counter))
                    self.tasks.rotate(-1)
            else:
                assert False
                
        elif self.env_name == "GoalReacher":
            if self.task_name == "wind" or self.task_name is None:
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    
                    self.env.task_id = self.current_task
        
                    self.tasks.rotate(-1)
                    
        # elif self.env_name == "GoalReacher":
        #     if self.task_name == "wind_power" or self.task_name == None:
        #         # Change wind power dynamically
        #         if self.counter % self.max_episode_len == 0:
        #             if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
        #                 self.true_labels.append(self.current_task)
        #             self.env.wind_power = 0.8  # Set wind power to the current task
        #             # print(f"SET WIND POWER TO TASK {self.current_task} AT STEP {self.counter}!")
        #             self.tasks.rotate(-1)
        #     elif self.task_name == "goal_radius":
        #         # Change goal radius dynamically
        #         if self.counter % self.max_episode_len == 0:
        #             if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
        #                 self.true_labels.append(self.current_task)
        #             self.env.goal_rad = self.tasks[0] * self.env.max_pos  # Adjust goal radius
        #             # print(f"SET GOAL RADIUS TO TASK {self.current_task} AT STEP {self.counter}!")
        #             self.tasks.rotate(-1)
                    
        elif self.env_name == "LunarLander":
            if self.task_name == "wind" or self.task_name is None:
                if self.counter % self.max_episode_len == 0:
                    if self.counter >= len(self.tasks) * self.max_episode_len * self.n_supervised_episodes:
                        self.true_labels.append(self.current_task)
                    
                    self.env.external_wind = self.lunar_winds[self.current_task]
        
                    self.tasks.rotate(-1)
                    
        else:
            assert False

       
        next_obs, reward, terminated, done, info = self.env.step(action)

      

        return next_obs, reward, terminated, done, info
    
    def reset(self, **kwargs):

        return self.env.reset(**kwargs)

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.trackbodyid = 0

    