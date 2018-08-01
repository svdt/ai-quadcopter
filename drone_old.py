import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class DroneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, '/Users/vdt/drone/drone.xml', 1)
        utils.EzPickle.__init__(self)

    def step(self, a):
        # a = 1+a
        # p_init = np.array([[0.1, 0.1, 0.01], [0.1, -0.1, 0.01], [-0.1, -0.1, 0.01], [-0.1, 0.1, 0.01]])
        # # a = a+1
        # # a = np.maximum(a,0)
        # q = self.state_vector()[3:7]
        # tr = np.array([2*(q[1]*q[3]-q[2]*q[0]), 2*(q[2]*q[3]+q[1]*q[0]), 1-2*(q[1]*q[1]+q[2]*q[2])])
        # p = np.ones(4)

        # p[0] += (p_init[0][2]-np.dot(p_init[0],tr))
        # p[1] += (p_init[1][2]-np.dot(p_init[1],tr))
        # p[2] += (p_init[2][2]-np.dot(p_init[2],tr))
        # p[3] += (p_init[3][2]-np.dot(p_init[3],tr))
        # a = p
        # p = p/np.linalg.norm(p)
        # print(np.linalg.norm([0.1, 0.1, 0.01]))
        # print(np.linalg.norm(p_init[0]))
        # print(np.linalg.norm(p[0]))
        # posbefore = self.get_body_com("quadrotor")[0]
        target = self.init_qpos[:3]
        self.do_simulation(a, self.frame_skip)
        # target =  np.array([0.0,0.0,1.0])
        # posafter = self.get_body_com("quadrotor")[0]
        state = self.state_vector()
        dist = np.linalg.norm(state[:3]-target)
        # a = (0.99+0.05*max((target[2]-state[2]),0))*0.73575*np.array([1.0,1.0,1.0,1.0])
        sinahalbe = np.sqrt(1-state[3]*state[3])
        if sinahalbe <= 1e-5:
            angle = 1.0
        else:
            angle = np.dot(state[4:7]/sinahalbe, np.array([0.0,0.0,1.0]))
        # reward = 2.0 + 1.0*(angle-1.0) - 5.0*np.linalg.norm(state[7:10]) #- 2.0*np.std(a)
        reward = 2.0*(angle-1.0) - 5*dist #- 2.0*np.std(a)

        notdone = np.isfinite(state).all() and angle >= 0.0 and dist < 4.5
        if dist > 1.5:
            reward -= 3.0
        done = not notdone
        if done:
            self.reset_model()
        ob = self._get_obs()
        return ob, reward, done, dict()

    def _get_obs(self):
        # return self.state_vector()
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        qpos = self.init_qpos# + self.np_random.uniform(size=self.model.nq, low=-0.01, high=.01)
        qvel = self.init_qvel# + self.np_random.randn(self.model.nv) * .01
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.6
        self.viewer.cam.elevation = -10
