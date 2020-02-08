import argparse
from utils import *
import numpy as np
import random
import tensorflow as tf
import time
import os
import logging
import gym
from gym import envs
from gym.spaces import Discrete, Box
import prettytensor as pt
from space_conversion import SpaceConversionEnv
import tempfile
import sys

parser = argparse.ArgumentParser("cpo")
parser.add_argument("--logfile", type=str, default="", help="path of log file")
parser.add_argument("--seed", type=int, default=123, help="")
parser.add_argument("--level", type=int, default=0, help="")
parser.add_argument("--max_episode_len", type=int, default=3, help="")
parser.add_argument("--threshold", type=float, default=-1.5, help="")
args=parser.parse_args()


class CPOAgent(object):

    config = dict2(**{
        "timesteps_per_batch": 1000,
        "max_pathlength": 10000,
        "max_kl": 0.01,
        "cg_damping": 0.1,
        "gamma": .95,
        "d0": args.threshold})#-0.667})

    def __init__(self, env):
        self.env = env
        if not isinstance(env.observation_space, Box) or \
           not isinstance(env.action_space, Discrete):
            print("Incompatible spaces.")
            exit(-1)
        print("Observation Space", env.observation_space)
        print("Action Space", env.action_space)
        self.session = tf.Session()
        self.end_count = 0
        self.train = True
        self.obs = obs = tf.placeholder(
            dtype, shape=[
                None, 2 * env.observation_space.shape[0] + env.action_space.n], name="obs")
        self.prev_obs = np.zeros((1, env.observation_space.shape[0]))
        self.prev_action = np.zeros((1, env.action_space.n))
        self.action = action = tf.placeholder(tf.int64, shape=[None], name="action")
        self.advant = advant = tf.placeholder(dtype, shape=[None], name="advant")
        # [C] cumulative constraints
        self.cumc = cumc = tf.placeholder(dtype, shape=[None], name="cumc")
        self.oldaction_dist = oldaction_dist = tf.placeholder(dtype, shape=[None, env.action_space.n], name="oldaction_dist")

        # Create neural network.
        # action_dist_n is softmax normalized
        action_dist_n, _ = (pt.wrap(self.obs).
                            fully_connected(64, activation_fn=tf.nn.tanh).
                            softmax_classifier(env.action_space.n))
        eps = 1e-6
        self.action_dist_n = action_dist_n
        N = tf.shape(obs)[0]
        p_n = slice_2d(action_dist_n, tf.range(0, N), action)
        oldp_n = slice_2d(oldaction_dist, tf.range(0, N), action)
        ratio_n = p_n / oldp_n
        self.ratio_n = ratio_n
        Nf = tf.cast(N, dtype)
        surr = -tf.reduce_mean(ratio_n * advant)  # Surrogate loss
        # [C] expected cumulative constraint
        self.c_surr = c_surr = -tf.reduce_mean(ratio_n * cumc)
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        #var_list = tf.trainable_variables()
        kl = tf.reduce_sum(oldaction_dist * tf.log((oldaction_dist + eps) / (action_dist_n + eps))) / Nf
        ent = tf.reduce_sum(-action_dist_n * tf.log(action_dist_n + eps)) / Nf

        self.losses = [surr, kl, ent]
        self.pg = flatgrad(surr, var_list)
        # [C] cg
        self.cpg = flatgrad(c_surr, var_list)
        # KL divergence where first arg is fixed
        # replace old->tf.stop_gradient from previous kl
        kl_firstfixed = tf.reduce_sum(tf.stop_gradient(
            action_dist_n) * tf.log(tf.stop_gradient(action_dist_n + eps) / (action_dist_n + eps))) / Nf
        grads = tf.gradients(kl_firstfixed, var_list)
        self.flat_tangent = tf.placeholder(dtype, shape=[None])
        #shapes = map(var_shape, var_list)
        shapes = [var_shape(x) for x in var_list]
        start = 0
        tangents = []
        for shape in shapes:
            size = np.prod(shape)
            param = tf.reshape(self.flat_tangent[start:(start + size)], shape)
            tangents.append(param)
            start += size
        gvp = [tf.reduce_sum(g * t) for (g, t) in zip(grads, tangents)]
        self.fvp = flatgrad(gvp, var_list)# H * tangents
        self.gf = GetFlat(self.session, var_list)
        self.sff = SetFromFlat(self.session, var_list)
        self.vf = VF(self.session)
        self.session.run(tf.initialize_all_variables())

    def act(self, obs, *args):
        obs = np.expand_dims(obs, 0)
        self.prev_obs = obs
        obs_new = np.concatenate([obs, self.prev_obs, self.prev_action], 1)

        action_dist_n = self.session.run(self.action_dist_n, {self.obs: obs_new})

        if self.train:
            action = int(cat_sample(action_dist_n)[0])
        else:
            action = int(np.argmax(action_dist_n))
        self.prev_action *= 0.0
        self.prev_action[0, action] = 1.0
        return action, action_dist_n, np.squeeze(obs_new)

    def learn(self):
        config = self.config
        start_time = time.time()
        numeptotal = 0
        i = 0
        #while True:
        ops = open(args.logfile, 'w')
        eprewards = list()
        epconstraints = list()
        while numeptotal < 30000:
            # Generating paths.
            print("Rollout")
            paths = rollout(
                self.env,
                self,
                config.max_pathlength,
                config.timesteps_per_batch)

            # Computing returns and estimating advantage function.
            for path in paths:
                path["baseline"] = self.vf.predict(path)
                path["returns"] = discount(path["rewards"], config.gamma)
                if "constraints" in path:
                    path["cumc"] = discount(path["constraints"], config.gamma)
                path["advant"] = path["returns"] - path["baseline"]

            # Updating policy.
            action_dist_n = np.concatenate([path["action_dists"] for path in paths])
            obs_n = np.concatenate([path["obs"] for path in paths])
            action_n = np.concatenate([path["actions"] for path in paths])
            baseline_n = np.concatenate([path["baseline"] for path in paths])
            returns_n = np.concatenate([path["returns"] for path in paths])
            c_returns_n = np.concatenate([path["cumc"] for path in paths])

            # Standardize the advantage function to have mean=0 and std=1.
            advant_n = np.concatenate([path["advant"] for path in paths])
            advant_n -= advant_n.mean()

            # Computing baseline function for next iter.

            advant_n /= (advant_n.std() + 1e-8)

            feed = {self.obs: obs_n,
                    self.action: action_n,
                    self.advant: advant_n,
                    self.oldaction_dist: action_dist_n,
                    self.cumc: c_returns_n}


            episoderewards = np.array(
                [path["rewards"].sum() for path in paths])
            episodeconstraints = np.array(
                [path["constraints"].sum() for path in paths])

            meansteprwd = np.array(
                [path["rewards"].mean() for path in paths])
            eprewards.extend(meansteprwd.tolist())
            epconstraints.extend(episodeconstraints.tolist())
            if len(eprewards) >= 500:
                ops.write("{}\t{}\n".format(np.mean(eprewards[:500]), -1.0*np.mean(epconstraints[:500])))
                del eprewards[:500]
                del epconstraints[:500]

            print("\n********** Iteration %i ************" % i)
            #if episoderewards.mean() > 0.95*(1.0-(-1.0*config.d0)) and episodeconstraints.mean() <= config.d0:
            #    self.train = False
            if not self.train:
                print("Episode mean: %f" % episoderewards.mean())
                self.end_count += 1
                if self.end_count > 100:
                    break
            if self.train:
                self.vf.fit(paths)
                thprev = self.gf()

                def fisher_vector_product(p):
                    feed[self.flat_tangent] = p
                    return self.session.run(self.fvp, feed) + config.cg_damping * p

                c_surr_val, b, ratio_n_val = self.session.run([self.c_surr, self.cpg, self.ratio_n],
                                                              feed_dict=feed)
                c = (-c_surr_val) - config.d0
                g = self.session.run(self.pg, feed_dict=feed)
                c_stepdir = conjugate_gradient(fisher_vector_product, -b)
                s = -b.dot(c_stepdir)
                assert s > 0, "invalid positive definite quadratic form"
                if c*c/s-config.max_kl > 0 and c < 0:
                    # just do TRPO
                    stepdir = conjugate_gradient(fisher_vector_product, -g)# s_{unscaled} = H^{-1}g
                    shs = .5 * stepdir.dot(fisher_vector_product(stepdir))# shs= 0.5 * s_{unscaled}^{T}Hs_{unscaled}
                    lm = np.sqrt(shs / config.max_kl)
                    fullstep = stepdir / lm
                    neggdotstepdir = -g.dot(stepdir)
                    def loss(th):
                        self.sff(th)
                        robj, cobj = self.session.run([self.losses[0], self.c_surr], feed_dict=feed)
                        if cobj > config.d0:
                            return 999999
                        return robj
                    theta = linesearch(loss, thprev, fullstep, neggdotstepdir / lm)
                elif c*c/s-config.max_kl > 0 and c > 0:
                    # infeasible
                    stepdir = c_stepdir
                    shs = .5 * stepdir.dot(fisher_vector_product(stepdir))# shs= 0.5 * s_{unscaled}^{T}Hs_{unscaled}
                    lm = np.sqrt(shs / config.max_kl)
                    fullstep = stepdir / lm
                    neggdotstepdir = -b.dot(stepdir)
                    def loss(th):
                        self.sff(th)
                        cobj = self.session.run(self.c_surr, feed_dict=feed)
                        return -cobj
                    theta = linesearch(loss, thprev, -fullstep, neggdotstepdir / lm)
                else:
                    # 2 branches
                    stepdir = conjugate_gradient(fisher_vector_product, -g)
                    q = -g.dot(stepdir)
                    r = -g.dot(c_stepdir)
                    lbd_a = np.sqrt((q-r*r/s)/(config.max_kl-c*c/s))
                    lbd_b = np.sqrt(q/config.max_kl)
                    if c < 0:
                        range_a = (0, r/c)
                        range_b = (max(0, r/c), 999999)
                    else:
                        range_a = (max(0, r/c), 999999)
                        range_b = (0, r/c)
                    def proj2range(val, leftb, rightb):
                        return max(leftb, min(rightb, val))
                    if range_a[0] < range_a[1]:
                        lbd_a_star = proj2range(lbd_a, range_a[0], range_a[1])
                        fa = (0.5 / lbd_a_star)*(r*r/s - q) + 0.5*lbd_a_star*(c*c/s - config.max_kl) - (r*c/s)
                    else:
                        lbd_a_star = None
                        fa = -999999
                    if range_b[0] < range_b[1]:
                        lbd_b_star = proj2range(lbd_b, range_b[0], range_b[1])
                        fb = -0.5 * (q/lbd_b_star + lbd_b_star*config.max_kl)
                    else:
                        lbd_b_star = None
                        fb = -999999
                    if fa >= fb:
                        lbd_star = lbd_a_star
                    else:
                        lbd_star = lbd_b_star
                    v_star = max(0.0, (lbd_star*c-r)/s)
                    #print("{}\t{}".format(fa, fb))
                    #print("{}\t{}\t{}\t{}".format(lbd_star, c, r, v_star))
                    #input()
                    #print(c_surr_val)
                    #print(ratio_n_val)
                    #input()
                    fullstep = -1.0 / lbd_star * conjugate_gradient(fisher_vector_product, -1.0*(g+v_star*b))
                    neggdotstepdir = -g.dot(fullstep)
                    def loss(th):
                        self.sff(th)
                        robj, cobj = self.session.run([self.losses[0], self.c_surr], feed_dict=feed)
                        if -cobj > config.d0:
                            return 999999
                        return robj
                    theta = linesearch(loss, thprev, fullstep, neggdotstepdir)
                self.sff(theta)

                surrafter, kloldnew, entropy = self.session.run(
                    self.losses, feed_dict=feed)
                if kloldnew > 2.0 * config.max_kl:
                    print("reset \\theta")
                    self.sff(thprev)

                stats = {}

                numeptotal += len(episoderewards)
                stats["Total number of episodes"] = numeptotal
                stats["Average sum of rewards per episode"] = episoderewards.mean()
                stats["Average sum of constraints perepisode"] = episodeconstraints.mean()
                stats["Entropy"] = entropy
                exp = explained_variance(np.array(baseline_n), np.array(returns_n))
                stats["Baseline explained"] = exp
                stats["Time elapsed"] = "%.2f mins" % ((time.time() - start_time) / 60.0)
                stats["KL between old and new distribution"] = kloldnew
                stats["Surrogate loss"] = surrafter
                for k, v in stats.items():
                    print(k + ": " + " " * (40 - len(k)) + str(v))
                if entropy != entropy:
                    exit(-1)
                if exp > 0.8:
                    self.train = False
            i += 1
        ops.close()

training_dir = tempfile.mkdtemp()
logging.getLogger().setLevel(logging.DEBUG)

#from env import Sim0
#env = Sim0(6)
from i2rs.envs import Sim0, Sim2, Sim3
if args.level == 0:
    env = Sim0(dims=6, num_arms=6, max_episode_len=3, seed=args.seed, negate_c=True)
elif args.level ==2:
    env = Sim2(dims=6, num_arms=6, max_episode_len=3, seed=args.seed, negate_c=True)
elif args.level ==3:
    env = Sim3(dims=6, num_arms=6, max_episode_len=3, seed=args.seed, negate_c=True)
env = SpaceConversionEnv(env, Box, Discrete)

agent = CPOAgent(env)
agent.learn()
