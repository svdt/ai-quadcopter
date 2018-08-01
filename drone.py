import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class DroneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, '/Users/vdt/drone/drone.xml', 1)
        utils.EzPickle.__init__(self)

    def toThrust(self,a,vel):
        density = 1.225
        diam = 0.123
        pitch = diam*np.pi*np.tan(10*np.pi/180)
        return density*0.25*np.pi*diam**2*((a*pitch)**2-a*pitch*vel)*(diam/(3.29546*pitch))**1.5

    def step(self, a):
        if len(a) == 4:
            thrust = self.toThrust(a,np.linalg.norm(self.sim.data.qvel[:3]))
            # print(thrust)
            a = np.append(a,thrust)
        self.do_simulation(a, self.frame_skip)
        state = self.state_vector()
        target = self.init_qpos[:3]
        dist = np.linalg.norm(state[:3]-target)
        sinahalbe = np.sqrt(1-state[3]*state[3])
        if sinahalbe <= 1e-6:
            angle = 1.0
        else:
            angle = self.sim.data.qpos[6]
        notdone = np.isfinite(state).all() and dist < 5.5# and angle >= -0.8
        done = not notdone
        if done:
            self.reset_model()
        ob = self._get_obs()
        return ob, 0, done, dict()

    def _get_obs(self):
        # return self.state_vector()
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=.01)
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .05
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 1
        self.viewer.cam.distance = self.model.stat.extent * 0.9
        self.viewer.cam.elevation = -10
