import threading
import tensorflow as tf
import numpy as np
import os
import shutil
import time
from config import *

#### entropy of logits
# def cat_entropy(logits):
#     a0 = logits
#     ea0 = tf.exp(a0)
#     z0 = tf.reduce_sum(ea0, 1, keepdims=True)
#     p0 = ea0 / z0
#     return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

N_S = observation_space.shape
if CONTINUOUS:
    N_A = action_space.shape
else:
    N_A = action_space.n

class ACNet(object):
    def __init__(self, scope):
        with tf.variable_scope(scope):
            self.s = tf.placeholder(tf.float32, [None, *N_S], 'S')
            # self.input = tf.transpose(self.s, [0, 2, 3, 1])
            self.input = self.s
            self.a_his = tf.placeholder(tf.float32, [None, *N_A], 'A')
            self.v_target = tf.placeholder(tf.float32, [None], 'Vtarget')

            # self.a_prob, self.v, self.ac_params = self._build_net(scope)
            self.mu, sigma, self.v, self.ac_params = self._build_net(scope)
            self.sigma = sigma + 1e-6

            # self.sample = tf.argmax(self.a_prob - tf.log(-tf.log(tf.random_uniform(tf.shape(self.a_prob)))), axis=-1)

            td = tf.subtract(self.v_target, self.v, name='TD_error')

            normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)

            with tf.name_scope('a_loss'):
                neglogpac = -tf.reduce_sum(normal_dist.log_prob(self.a_his), axis=-1)
                #neglogpac = 0.5 * tf.reduce_sum(tf.square((self.a_his - self.mu)) / self.sigma, axis=-1) + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(self.a_his)[-1]) + tf.reduce_sum(tf.log(self.sigma), axis=-1)
                a_loss = tf.reduce_mean(tf.stop_gradient(td) * neglogpac)#+0.05*tf.reduce_mean(self.sigma)
                entropy = tf.reduce_mean(tf.reduce_sum(normal_dist.entropy(), axis=-1))
                # entropy = tf.reduce_mean(cat_entropy(self.a_prob))
                # c_loss = tf.reduce_sum(tf.square(td), axis=-1)
                c_loss = tf.reduce_mean(tf.square(td), axis=-1)
                self.ac_loss = a_loss - ENTROPY_BETA*entropy + 0.5*c_loss

            with tf.name_scope('choose_a'):  # use local params to choose action
                # self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1]) # sample a action from distribution
                self.A = tf.squeeze(normal_dist.sample(1), axis=0)

            with tf.name_scope('local_grad'):
                self.ac_grads = tf.gradients(self.ac_loss, self.ac_params)
                if max_grad_norm is not None:
                    self.ac_grads, grad_norm = tf.clip_by_global_norm(self.ac_grads, max_grad_norm)

            tf.summary.scalar("mean-mean", tf.reduce_mean(self.mu))
            tf.summary.scalar("max-std", tf.reduce_max(self.sigma))
            tf.summary.scalar("actor-critic-loss", self.ac_loss)
            tf.summary.scalar("actor-critic-max-grad-norm", tf.reduce_max([tf.norm(g) for g in self.ac_grads]))
            self.mean_episode_reward = tf.Variable(0.)
            tf.summary.scalar("episode reward", self.mean_episode_reward)
            self.summary = tf.summary.merge_all(scope=scope)
            self.opt_ac = tf.train.RMSPropOptimizer(learning_rate=LR_AC, name='RMSPropAC', decay=ALPHA, epsilon=EPS)

            with tf.name_scope('update'):
                self.update_ac_op = self.opt_ac.apply_gradients(zip(self.ac_grads, self.ac_params))

    def _build_net(self, scope):
        with tf.variable_scope('actor-critic'):
            w_init = tf.orthogonal_initializer()
            # scaled_images = tf.cast(self.input, tf.float32) / 255.
            # # activ = tf.nn.relu
            # h = tf.layers.conv2d(scaled_images, 16,(8,8),strides=(4, 4),activation=tf.nn.relu,kernel_initializer=w_init,name='AC_cnn1')
            # h = tf.layers.conv2d(h, 32,(3,3),strides=(1, 1),activation=tf.nn.relu,kernel_initializer=w_init,name='AC_cnn2')
            # h = tf.layers.conv2d(h, 32,(2,2),strides=(1, 1),activation=tf.nn.relu,kernel_initializer=w_init,name='AC_cnn3')
            # flat = tf.layers.flatten(h)
            flat = self.input
            h = tf.layers.dense(flat, 256,activation=tf.nn.relu,kernel_initializer=w_init,name='AC_fc1')
            h = tf.layers.dense(h, 256,activation=tf.nn.relu,kernel_initializer=w_init,name='AC_fc2')
            # h = tf.layers.dense(h, 512,activation=tf.nn.relu,kernel_initializer=w_init,name='AC_fc3')

            h_pi = tf.layers.dense(h, 128,activation=tf.nn.relu,kernel_initializer=w_init,name='AC_fcpi')
            # h_pi = tf.layers.dense(h_pi, 128,activation=tf.nn.relu,kernel_initializer=w_init,name='AC_fcpi2')
            # a_prob = tf.layers.dense(h_pi, N_A,activation=tf.nn.softmax,name='AC_pi')
            mu = tf.layers.dense(h_pi, *N_A,activation=tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(h_pi, *N_A,activation=tf.nn.softplus, kernel_initializer=w_init, name='sigma')

            h_v = tf.layers.dense(h, 128,activation=tf.nn.relu,kernel_initializer=w_init,name='AC_fcv')
            # h_v = tf.layers.dense(h_v, 128,activation=tf.nn.relu,kernel_initializer=w_init,name='AC_fcv2')
            v = tf.layers.dense(h_v, 1,name='AC_v')[:,0]

        with tf.variable_scope('actor-critic'):
            total_parameters = 0
            for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor-critic'):
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            print('Model params: ',total_parameters)
            ac_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor-critic')
        return mu, sigma, v, ac_params

    def update(self, feed_dict):  # run by a local
        SESS.run([self.update_ac_op], feed_dict)  # local grads applies to global net

    def choose_action(self, s):  # run by a local
        # return SESS.run([self.sample], feed_dict={self.s: s})[0]
        # prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s})
        # actions = []
        # for pw in prob_weights:
        #     actions.append(np.random.choice(range(len(pw)),p=pw.ravel()))  # select action w.r.t the actions prob
        # return actions
        return SESS.run(self.A, feed_dict={self.s: s})


class Worker(object):
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        with tf.device("/cpu:0"):
            self.ACs = {}
            self.state = {}
            # Create worker
            for agent in self.agents:
                self.ACs[agent] = ACNet(agent)
                # self.state[agent] = self.env.reset(agent)
                self.state[agent] = self.env.reset(agent)

    def work(self):
        global EPISODE
        ep_r, buffer_s, buffer_a, buffer_v, buffer_r, buffer_done = {}, {}, {}, {}, {}, {}
        for agent in self.agents:
            buffer_s[agent], buffer_a[agent], buffer_r[agent], buffer_v[agent], buffer_done[agent] = [], [], [], [], []
            ep_r[agent] = np.zeros(len(self.state[agent]))

        for ep_t in range(UPDATE_ITER):
            for agent in self.agents:
                a_a = self.ACs[agent].choose_action(self.state[agent])
                # a_s, a_r, a_done, a_info = self.env.step(agent, a_a)
                a_s, a_r, a_done, a_info = self.env.step(agent, a_a)
                if RENDER:
                    env.render()

                ep_r[agent] += a_r
                buffer_s[agent].append(np.copy(self.state[agent]))
                buffer_a[agent].append(a_a)
                buffer_r[agent].append(a_r)
                buffer_done[agent].append(list(a_done))
                self.state[agent] = list(a_s)

        if not TESTING:

            # get values for last state
            for agent in self.agents:
                #batch of steps to batch of rollouts
                # buffer_a[agent] = np.asarray(buffer_a[agent], dtype=np.float32).swapaxes(1, 0).flatten()
                buffer_a[agent] = np.vstack(np.asarray(buffer_a[agent], dtype=np.float32).swapaxes(1, 0))

                R = SESS.run(self.ACs[agent].v, {self.ACs[agent].s: self.state[agent]})
                buffer_s[agent] = np.vstack(np.asarray(buffer_s[agent], dtype=np.float32).swapaxes(1, 0))

                rewards = []
                for r,done in zip(reversed(buffer_r[agent]),reversed(buffer_done[agent])):
                    R = r + GAMMA*R*np.logical_not(done)
                    rewards.append(R)
                rewards = np.asarray(rewards[::-1]).swapaxes(1, 0).flatten()
                feed_dict = {
                    self.ACs[agent].s: buffer_s[agent],
                    self.ACs[agent].a_his: buffer_a[agent],
                    self.ACs[agent].v_target: rewards,
                    self.ACs[agent].mean_episode_reward: np.mean(ep_r[agent])
                }
                self.ACs[agent].update(feed_dict)

                if OUTPUT_GRAPH:
                    GLOBAL_WRITER.add_summary(SESS.run(self.ACs[agent].summary, feed_dict=feed_dict),EPISODE)
                    GLOBAL_WRITER.flush()

def render_env(env):
    while RENDER:
        env.render()
        time.sleep(0.2)

if __name__ == "__main__":
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=4,
        intra_op_parallelism_threads=4)
    tf_config.gpu_options.allocator_type = 'BFC'
    SESS = tf.Session(config=tf_config)

    worker = Worker(env, agents)
    SESS.run(tf.global_variables_initializer())

    SAVER = tf.train.Saver()
    if TESTING or CONTINUE:
        SAVER.restore(SESS, SAVE_DIR)
        print("Model restored.")

    if not TESTING and OUTPUT_GRAPH:
        FNAME = 'run'
        RUNS = [int(x.split(' ')[1]) for x in os.listdir(LOG_DIR) if os.path.isdir(LOG_DIR+'/'+x)]
        N = 0 if len(RUNS) == 0 else np.max(RUNS)
        GLOBAL_WRITER = tf.summary.FileWriter(LOG_DIR+'/'+FNAME+' '+str(N+1), SESS.graph)
        for R in RUNS:
            DELDIR = LOG_DIR+'/'+FNAME+' '+str(R)
            if R < N+1-KEEP_N and os.path.exists(DELDIR):
                shutil.rmtree(DELDIR)

    # if RENDER:
    #     t = threading.Thread(target=render_env, args=(env,))
    #     t.start()
    while EPISODE < MAX_GLOBAL_EP:
        try:
            worker.work()
            EPISODE += 1
        except KeyboardInterrupt:
            break
    RENDER = False
    if not TESTING:
        print("Model saved in path: %s" % SAVER.save(SESS, SAVE_DIR))
