#!/usr/bin/env python

import pickle
import tensorflow as tf
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
import os
import tf_util
import argparse
import load_policy


def load_data(data_file):
    with open(data_file, 'rb') as f:
        data = pickle.loads(f.read())
    return data


class BehaviorCloning:
    def __init__(self, env, envname, args):
        self.args = args
        self.envname = envname
        self.model = Sequential()
        self.model.add(Dense(units=128,
                             input_dim=env.observation_space.shape[0],
                             activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=env.action_space.shape[0]))
        self.model.compile(loss='mse', optimizer='adam')

    def train(self, data_file):
        expert_data = load_data(data_file)
        obs_data = np.array(expert_data['observations'])
        obs_data = obs_data.reshape(obs_data.shape[0], obs_data.shape[1])
        act_data = np.array(expert_data['actions'])
        act_data = act_data.reshape(act_data.shape[0], act_data.shape[2])
        self.model.fit(obs_data, act_data, epochs=self.args.epochs, verbose=1)
        # score = self.model.evaluate()
        self.model.save('bc/%s.model' % self.envname)
        return self.model


class AlternativeBC:
    def __init__(self, env, envname, args):
        self.args = args
        self.envname = envname
        self.model = Sequential()
        self.model.add(Dense(units=128,
                             input_dim=env.observation_space.shape[0],
                             activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=env.action_space.shape[0]))
        self.model.compile(loss='mse', optimizer='adam')

    def train(self, data_file):
        expert_data = load_data(data_file)
        obs_data = np.array(expert_data['observations'])
        obs_data = obs_data.reshape(obs_data.shape[0], obs_data.shape[1])
        act_data = np.array(expert_data['actions'])
        act_data = act_data.reshape(act_data.shape[0], act_data.shape[2])
        self.model.fit(obs_data, act_data, epochs=self.args.epochs, verbose=1)
        # score = self.model.evaluate()
        self.model.save('alternative_bc/%s.model' % self.envname)
        return self.model


class Dagger:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.model = Sequential()
        self.model.add(Dense(units=128,
                             input_dim=env.observation_space.shape[0],
                             activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=env.action_space.shape[0]))

        self.model.compile(loss='mse', optimizer='adam')

        print('loading and building expert policy')
        self.policy_fn = load_policy.load_policy(args.expert_policy_file)
        print('loaded and built')

    def train(self, data_file):
        expert_data = load_data(data_file)
        obs_data = np.array(expert_data['observations'])
        act_data = np.array(expert_data['actions'])
        with tf.Session():
            tf_util.initialize()
            # Dagger loop
            for i in range(self.args.iteration):
                print('obs_data.shape = %s' % str(obs_data.shape))
                print('act_data.shape = %s' % str(act_data.shape))
                # Train on D
                obs_data_reshape = obs_data.reshape(obs_data.shape[0], obs_data.shape[1])
                act_data_reshape = act_data.reshape(act_data.shape[0], act_data.shape[2])
                self.model.fit(obs_data_reshape, act_data_reshape, epochs=10, verbose=1)
                # run model and label actions with expert policy
                new_obs_data, new_act_data = self.run_and_label()
                # add new obs and act into dataset - aggregate
                obs_data = np.concatenate((obs_data, new_obs_data))
                act_data = np.concatenate((act_data, new_act_data))

            self.model.save('dagger/%s.model' % self.args.envname)

    def run_and_label(self):
        # max_steps = args.max_timesteps or env.spec.timestep_limit
        max_steps = self.env.spec.timestep_limit
        num_rollout = 100

        new_obs_data = []
        new_act_data = []
        for i in range(num_rollout):
            print('iter', i)
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                labeled_act = self.policy_fn(obs[None, :])
                new_obs_data.append(obs)
                new_act_data.append(labeled_act)

                obs = np.array(obs)
                obs = obs.reshape(1, len(obs))
                action = self.model.predict(obs, verbose=0)

                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1
                if self.args.render:
                    self.env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break

        new_obs_data = np.array(new_obs_data)
        new_act_data = np.array(new_act_data)

        return new_obs_data, new_act_data


class AlternativeDagger:
    def __init__(self, env, args):
        self.args = args
        self.env = env
        self.model = Sequential()
        self.model.add(Dense(units=128,
                             input_dim=env.observation_space.shape[0],
                             activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dense(units=env.action_space.shape[0]))

        self.model.compile(loss='mse', optimizer='adam')

        print('loading and building expert policy')
        self.policy_fn = load_policy.load_policy(args.expert_policy_file)
        print('loaded and built')

    def train(self, data_file):
        expert_data = load_data(data_file)
        obs_data = np.array(expert_data['observations'])
        act_data = np.array(expert_data['actions'])
        with tf.Session():
            tf_util.initialize()
            # Dagger loop
            for i in range(self.args.iteration):
                print('obs_data.shape = %s' % str(obs_data.shape))
                print('act_data.shape = %s' % str(act_data.shape))
                # Train on D
                obs_data_reshape = obs_data.reshape(obs_data.shape[0], obs_data.shape[1])
                act_data_reshape = act_data.reshape(act_data.shape[0], act_data.shape[2])
                self.model.fit(obs_data_reshape, act_data_reshape, epochs=10, verbose=1)
                # run model and label actions with expert policy
                new_obs_data, new_act_data = self.run_and_label()
                # add new obs and act into dataset - aggregate
                obs_data = np.concatenate((obs_data, new_obs_data))
                act_data = np.concatenate((act_data, new_act_data))

            self.model.save('alternative_dagger/%s.model' % self.args.envname)

    def run_and_label(self):
        # max_steps = args.max_timesteps or env.spec.timestep_limit
        max_steps = self.env.spec.timestep_limit
        num_rollout = 100

        new_obs_data = []
        new_act_data = []
        for i in range(num_rollout):
            print('iter', i)
            obs = self.env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                labeled_act = self.policy_fn(obs[None, :])
                new_obs_data.append(obs)
                new_act_data.append(labeled_act)

                obs = np.array(obs)
                obs = obs.reshape(1, len(obs))
                action = self.model.predict(obs, verbose=0)

                obs, r, done, _ = self.env.step(action)
                totalr += r
                steps += 1
                if self.args.render:
                    self.env.render()
                if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                if steps >= max_steps:
                    break

        new_obs_data = np.array(new_obs_data)
        new_act_data = np.array(new_act_data)

        return new_obs_data, new_act_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--expert_policy_file', type=str)
    parser.add_argument('--dagger', action='store_true')
    parser.add_argument('--bc', action='store_true')
    parser.add_argument('--alternative', type=str)
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int,
                        help='Number of expert roll outs')

    args = parser.parse_args()

    import gym
    env = gym.make(args.envname)

    if args.num_rollouts is None:
        if args.bc:
            # train behavior cloning
            bc = BehaviorCloning(env, args.envname, args)
            bc.train(data_file=args.data_file)
        if args.dagger:
            dagger = Dagger(env, args)
            dagger.train(data_file=args.data_file)
        if args.alternative == 'bc':
            bc = AlternativeBC(env, args.envname, args)
            bc.train(data_file=args.data_file)
        if args.alternative == 'dagger':
            dagger = AlternativeDagger(env, args)
            dagger.train(data_file=args.data_file)
    else:
        with tf.Session():
            tf_util.initialize()

            max_steps = args.max_timesteps or env.spec.timestep_limit

            # generate rollouts stats
            # load model
            if args.bc:
                model = load_model('bc/%s.model' % args.envname)
            elif args.dagger:
                model = load_model('dagger/%s.model' % args.envname)
            else:
                if args.alternative == 'bc':
                    model = load_model('alternative_bc/%s.model' % args.envname)
                if args.alternative == 'dagger':
                    model = load_model('alternative_dagger/%s.model' % args.envname)

            returns = []
            observations = []
            actions = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    obs = np.array(obs)
                    obs = obs.reshape(1, len(obs))
                    action = model.predict(obs, verbose=0)
                    observations.append(obs)
                    # print('action shape = %s' % str(action.shape))
                    # action = action.reshape(env.action_space.shape)
                    actions.append(action)
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

                if (i+1) % 20 == 0:
                    print('returns', returns)
                    print('mean return', np.mean(returns))
                    print('std of return', np.std(returns))

                    if args.stats:
                        out_dir = 'bc' if args.bc else 'dagger'
                        with open(os.path.join(out_dir, '%s_%s.rollout'
                                % (args.envname, str(args.num_rollouts))), 'a') as f:
                            f.write(str(i+1) + '\n')
                            f.write(str(np.mean(returns)) + '\n')
                            f.write(str(np.std(returns)) + '\n\n')


if __name__ == '__main__':
    main()

