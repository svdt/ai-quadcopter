import numpy as np
import math, time
from drone import DroneEnv

env = DroneEnv()
dt = env.sim.model.opt.timestep

g = 9.81
m = 0.3

def toEuler(q):
    w,x,y,z = q
    ysqr = y * y
    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.atan2(t0, t1)
    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.asin(t2)
    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.atan2(t3, t4)
    return np.array([X, Y, Z])

class Controller:
    def __init__(self,Kp,Ki,Kd,Kp_angle,Ki_angle,Kd_angle):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Kp_angle = Kp_angle
        self.Ki_angle = Ki_angle
        self.Kd_angle = Kd_angle
        self.motor_limits = [0,800]
        self.reset()

    def control(self,pos,vel):
        e = self.target# - pos[:3]
        self.integral += dt*e
        derivative = (self.target-self.old_target)/dt-vel[:3]
        dest_dot = self.Kp*e + self.Kd*derivative + self.Ki*self.integral
        throttle = dest_dot[2]
        yaw = pos[5]
        dest_angle = np.zeros(3)
        dest_angle[:2] = np.array([[np.sin(yaw),-np.cos(yaw)],[np.cos(yaw),np.sin(yaw)]]).dot(dest_dot[:2])
        dest_angle[2] = pos[5] + self.yaw_target
        e_angle = dest_angle - pos[3:]
        self.integral_angle += dt*e_angle
        derivative_angle = 0*(dest_angle-self.dest_angle_old)/dt-vel[3:]
        dest_dot_angle = self.Kp_angle*e_angle + self.Kd_angle*derivative_angle + self.Ki_angle*self.integral_angle
        a = throttle*np.ones(4) + np.array([ dest_dot_angle[0]+dest_dot_angle[2],
                                            -dest_dot_angle[1]-dest_dot_angle[2],
                                            -dest_dot_angle[0]+dest_dot_angle[2],
                                             dest_dot_angle[1]-dest_dot_angle[2]])
        self.integral = np.where(np.abs(self.integral)>1,np.zeros(3),self.integral)
        self.integral_angle = np.where(np.abs(self.integral_angle)>1,np.zeros(3),self.integral_angle)
        print(a)
        # print("e:",e)
        self.old_target = self.target
        self.dest_angle_old = dest_angle
        self.yaw_target = self.yaw_target - dt*vel[5]
        self.target = self.target - dt*vel[:3]
        # print("ie:",(self.target-self.old_target)/dt)
        return np.clip(a,self.motor_limits[0],self.motor_limits[1])

    def addTarget(self,d,amount):
        self.target[d] = amount

    def addYaw_target(self,amount):
        self.yaw_target = amount

    def stop(self,d=-1):
        if d == -1:
            self.yaw_target = 0
        else:
            self.target[d] = 0

    def reset(self):
        self.integral = np.zeros(3)
        self.integral_angle = np.zeros(3)
        self.target = np.zeros(3)
        self.old_target = np.zeros(3)
        self.target[2] = 2
        self.yaw_target = 0
        self.dest_angle_old = np.zeros(3)

Kp = np.array([250,250,15000])
Ki = np.array([0.1,0.1,4.5])
Kd = np.array([990,990,8000])

Kp_angle = np.array([1,1,120])
Ki_angle = np.array([1,1,3])
Kd_angle = np.array([500,500,100])

controller = Controller(Kp,Ki,Kd,Kp_angle,Ki_angle,Kd_angle)

from threading import Thread, Event
import keyboard

def KeyboardListen(cont,env):
    esc = False
    amount = 1
    while not esc:
        key = keyboard.read_event()
        if key.name == 'esc':
            esc = True
        if key.name == 'up':
            cont.addTarget(2,amount)
            if key.event_type == 'up':
                cont.stop(2)
        if key.name == 'down':
            cont.addTarget(2,-amount)
            if key.event_type == 'up':
                cont.stop(2)
        if key.name == 'left':
            cont.addTarget(0,-amount)
            if key.event_type == 'up':
                cont.stop(0)
        if key.name == 'right':
            cont.addTarget(0,amount)
            if key.event_type == 'up':
                cont.stop(0)
        if key.name == 'w':
            cont.addTarget(1,amount)
            if key.event_type == 'up':
                cont.stop(1)
        if key.name == 's':
            cont.addTarget(-amount)
            if key.event_type == 'up':
                cont.stop(1)
        if key.name == 'y':
            cont.addYaw_target(2*amount)
            if key.event_type == 'up':
                cont.stop()
        if key.name == 'x':
            cont.addYaw_target(-2*amount)
            if key.event_type == 'up':
                cont.stop()
        if key.name == 'q':
            env.reset()
            cont.reset()

t = Thread(target=KeyboardListen, args=(controller,env))
t.start()

render = True
n = 100
for i in range(n):
    env.reset()
    done = False
    t = 0
    controller.reset()
    while not done:
        qpos_new = env.sim.data.qpos
        qvel_new = env.sim.data.qvel
        # PID controller
        pos = np.append(env.sim.data.qpos[:3],toEuler(env.sim.data.qpos[3:7]))
        vel = env.sim.data.qvel[:6]
        a = controller.control(pos,vel)
        s_, r, done, _ = env.step(a)
        if render:
            env.render()
        t += dt


# def KeyControl(key):
#     print(key)
#
# lis = keyboard.Listener(on_press=KeyControl)
# lis.start()
# lis.join()

# except Exception as e:
#     print(e)
#     print("End simulation..")
