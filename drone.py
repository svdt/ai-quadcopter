import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class DroneEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, '/Users/vdt/drone/drone.xml', 1)
        utils.EzPickle.__init__(self)
#     i_term = np.zeros(3)
#     params = {'Motor_limits':[4000,9000],
#                 'Tilt_limits':[-10,10],
#                 'Yaw_Control_Limits':[-900,900],
#                 'Z_XY_offset':500,
#                 'Linear_PID':{'P':[2000,2000,7000],'I':[0.25,0.25,4.5],'D':[50,50,5000]},
#                 'Linear_To_Angular_Scaler':[1,1,0],
#                 'Yaw_Rate_Scaler':0.18,
#                 'Angular_PID':{'P':[22000,22000,1500],'I':[0,0,1.2],'D':[12000,12000,0]},
#                 }
#     self.LINEAR_P = params['Linear_PID']['P']
#     self.LINEAR_I = params['Linear_PID']['I']
#     self.LINEAR_D = params['Linear_PID']['D']
#     self.LINEAR_TO_ANGULAR_SCALER = params['Linear_To_Angular_Scaler']
#     self.YAW_RATE_SCALER = params['Yaw_Rate_Scaler']
#     self.ANGULAR_P = params['Angular_PID']['P']
#     self.ANGULAR_I = params['Angular_PID']['I']
#     self.ANGULAR_D = params['Angular_PID']['D']
#
# def getEulerState(self):
#     state = self.state_vector()
#     pos = state[:3]
#     vel = state[7:10]
#     qpos = state[3:7]
#     qvel = state[10:14]
#     return np.concatenate(state[:3], state[8:])
#
# def getControlTarget(self,target):
#     [x,y,z,x_dot,y_dot,z_dot,theta,phi,gamma,theta_dot,phi_dot,gamma_dot] = self.get_state(self.quad_identifier)
#     pos, vel, angle, angle_dot = self.getEulerState()
#     error = target-pos
#
#     self.i_term += self.LINEAR_I*error
#     dest_dot = self.LINEAR_P*error - self.LINEAR_D*vel + self.i_term
#     throttle = np.maximum(dest_dot[3],0)
#     dest_angle = np.array([self.LINEAR_TO_ANGULAR_SCALER[0]*(dest_dot[0]*math.sin(angle[3])-dest_dot[1]*math.cos(angle[3])), \
#                             self.LINEAR_TO_ANGULAR_SCALER[1]*(dest_dot[0]*math.cos(angle[3])+dest_dot[1]*math.sin(angle[3]))])
#     dest_gamma = self.yaw_target
#
#     dest_angle = np.clip(dest_angle,-10,10)
#     angle_error = dest_angle-angle
#     gamma_dot_error = (self.YAW_RATE_SCALER*self.wrap_angle(dest_gamma-angle[3])) - angle_dot[0]
#     angle_error = np.append(angle_error, gamma_dot_error)
#     self.i_term += self.ANGULAR_I*angle_error
#     x_val = self.ANGULAR_P[0]*(theta_error) + self.ANGULAR_D[0]*(-theta_dot) + self.thetai_term
#     y_val = self.ANGULAR_P[1]*(phi_error) + self.ANGULAR_D[1]*(-phi_dot) + self.phii_term
#     z_val = self.ANGULAR_P[2]*(gamma_dot_error) + self.gammai_term
#     z_val = np.clip(z_val,self.YAW_CONTROL_LIMITS[0],self.YAW_CONTROL_LIMITS[1])
#     m1 = throttle + x_val + z_val
#     m2 = throttle + y_val - z_val
#     m3 = throttle - x_val + z_val
#     m4 = throttle - y_val - z_val
#     return np.array([m1,m2,m3,m4])

    def step(self, a):
        a = 1+a
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
        # target = self.init_qpos[:3]
        self.do_simulation(a, self.frame_skip)
        target =  np.array([0.0,0.0,1.0])
        # posafter = self.get_body_com("quadrotor")[0]
        state = self.state_vector()
        dist = np.linalg.norm(state[:3]-target)
        # a = (0.99+0.05*max((target[2]-state[2]),0))*0.73575*np.array([1.0,1.0,1.0,1.0])
        sinahalbe = np.sqrt(1-state[3]*state[3])
        if sinahalbe <= 1e-3:
            angle = 1.0
        else:
            angle = np.dot(state[4:7]/sinahalbe, np.array([0.0,0.0,1.0]))
        # reward = 2.0 + 1.0*(angle-1.0) - 5.0*np.linalg.norm(state[7:10]) #- 2.0*np.std(a)
        reward = 2.0*(angle-1.0) - 5*dist #- 2.0*np.std(a)

        notdone = np.isfinite(state).all() and angle >= 0.0 and dist < 1.5
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
        qpos = self.init_qpos #+ self.np_random.uniform(size=self.model.nq, low=0.1, high=.2)
        qvel = self.init_qvel #+ self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5
