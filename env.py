import tensorflow as tf
import sys
import numpy as np
import argparse
import pybullet as p
import time

# class DroneEnv(object):
#
# 	def __init__(self):
cid = p.connect(p.SHARED_MEMORY)
if (cid<0):
	cid = p.connect(p.GUI) #DIRECT is much faster, but GUI shows the running gait
gravity = 9.81
p.setGravity(0,0,-gravity)
p.setPhysicsEngineParameter(fixedTimeStep=1.0/60., numSolverIterations=5, numSubSteps=2)
#this mp4 recording requires ffmpeg installed
#mp4log = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4,"humanoid.mp4")

p.loadSDF("/usr/local/lib/python3.6/site-packages/pybullet_data/stadium.sdf")
#p.loadURDF("plane.urdf")

objs = p.loadMJCF("/Users/vdt/drone/drone2.xml",flags = p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
drone  = objs[0]

class Dummy:
	pass
dummy = Dummy()
dummy.initial_z = None

def current_relative_position(jointStates, drone, j, lower, upper):
	#print("j")
	#print(j)
	#print (len(jointStates))
	#print(j)
	temp  = jointStates[j]
	pos = temp[0]
	vel = temp[1]
	#print("pos")
	#print(pos)
	#print("vel")
	#print(vel)
	pos_mid = 0.5 * (lower + upper);
	return (
		2 * (pos - pos_mid) / (upper - lower),
		0.1 * vel
		)

def collect_observations(drone):
	#print("ordered_joint_indices")
	#print(ordered_joint_indices)

	jointStates = p.getJointStates(drone,ordered_joint_indices)
	j = np.array([current_relative_position(jointStates, drone, *jtuple) for jtuple in ordered_joints]).flatten()
	#print("j")
	#print(j)
	body_xyz, (qx, qy, qz, qw) = p.getBasePositionAndOrientation(drone)
	#print("body_xyz")
	#print(body_xyz, qx,qy,qz,qw)
	z = body_xyz[2]
	dummy.distance = body_xyz[0]
	if dummy.initial_z==None:
		dummy.initial_z = z
	(vx, vy, vz), _ = p.getBaseVelocity(drone)
	more = np.array([z-dummy.initial_z, 0.1*vx, 0.1*vy, 0.1*vz, qx, qy, qz, qw])
	# rcont = p.getContactPoints(drone, -1, right_foot, -1)
	# #print("rcont")
	# #print(rcont)
	# lcont = p.getContactPoints(drone, -1, left_foot, -1)
	# #print("lcont")
	# #print(lcont)
	# feet_contact = np.array([len(rcont)>0, len(lcont)>0])
	# return np.clip( np.concatenate([more] + [j] + [feet_contact]), -5, +5)
	return np.clip( np.concatenate([more] + [j]), -5, +5)

mass = p.getDynamicsInfo(drone, -1)[0]
initPos, initOrn = p.getBasePositionAndOrientation(drone)

p.resetBasePositionAndOrientation(drone, initPos+np.random.uniform(size=3, low=-.1, high=.1), initOrn+np.random.uniform(size=4, low=-.1, high=.1))

frame = 0
while 1:
	# print(collect_observations(drone))
	actions = mass*gravity/4*np.array([1,1,1,1])
	dronePos, droneOrn = p.getBasePositionAndOrientation(drone)
	# angle = 2*np.acos(droneOrn[-1])
	sinahalbe = np.sqrt(1-droneOrn[-1]*droneOrn[-1])
	if sinahalbe <= 1e-8:
		orientation = np.array([0.0,0.0,1.0])
		angle = 1.0
	else:
		orientation = droneOrn[:-1]/sinahalbe
		angle = np.dot(orientation, np.array([0.0,0.0,1.0]))

	print(orientation)
	p.applyExternalForce(drone,-1, actions[0]*orientation, np.array([0.1, 0.1, 0.01]), flags=p.LINK_FRAME)
	p.applyExternalForce(drone,-1, actions[1]*orientation, np.array([0.1, -0.1, 0.01]), flags=p.LINK_FRAME)
	p.applyExternalForce(drone,-1, actions[2]*orientation, np.array([-0.1, -0.1, 0.01]), flags=p.LINK_FRAME)
	p.applyExternalForce(drone,-1, actions[3]*orientation, np.array([-0.1, 0.1, 0.01]), flags=p.LINK_FRAME)
	p.stepSimulation()
	time.sleep(0.2)
	# distance=5
	# yaw = 0
	# p.resetDebugVisualizerCamera(distance,yaw,-20,dronePos);
	frame += 1

	if frame==1000: break
	time.sleep(0.01)
# def demo_run():
# 	sess = tf.InteractiveSession(config=tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1, device_count={ "GPU": 0 }))
# 	pi = SmallReactivePolicy("pi", np.zeros((44,)), np.zeros((17,)))
# 	pi.load_weights()
# 	t1 = time.time()
# 	# timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "humanoidTimings.json")
#
# 	frame = 0
# 	while 1:
# 		obs = collect_observations(drone)
#
# 		# actions = pi.act(obs)
# 		actions = [10,10,10,10]
#
#
# 		#print(" ".join(["%+0.2f"%x for x in obs]))
# 		#print("Motors")
# 		#print(motors)
#
# 		#for m in range(len(motors)):
# 			#print("motor_power")
# 			#print(motor_power[m])
# 			#print("actions[m]")
# 			#print(actions[m])
# 		#p.setJointMotorControl2(human, motors[m], controlMode=p.TORQUE_CONTROL, force=motor_power[m]*actions[m]*0.082)
# 		#p.setJointMotorControl2(human1, motors[m], controlMode=p.TORQUE_CONTROL, force=motor_power[m]*actions[m]*0.082)
#
# 		forces = [0.] * len(motors)
# 		for m in range(len(motors)):
# 			forces[m] = motor_power[m]*actions[m]*0.082
# 		p.setJointMotorControlArray(drone, motors,controlMode=p.TORQUE_CONTROL, forces=forces)
#
# 		p.stepSimulation()
# 		time.sleep(0.01)
# 		distance=5
# 		yaw = 0
# 		dronePos, droneOrn = p.getBasePositionAndOrientation(drone)
# 		p.resetDebugVisualizerCamera(distance,yaw,-20,dronePos);
# 		frame += 1
#
# 		if frame==1000: break
# 	t2 = time.time()
# 	print("############################### distance = %0.2f meters" % dummy.distance)
# 	print("############################### FPS = ", 1000/ (t2-t1))
# 	#print("Starting benchmark")
# 	#logId = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS,"pybullet_humanoid_timings.json")
# 	#p.stopStateLogging(logId)
# 	print("ended benchmark")
# 	print(frame)
# 	# p.stopStateLogging(timinglog)
