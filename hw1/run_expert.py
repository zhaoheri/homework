#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v2.pkl Humanoid-v2 --render --num_rollouts 100

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--output_rollout', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')

    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        print('evn.observation_space = %s' % str(env.observation_space))
        print('evn.action_space = %s' % str(env.action_space))
        max_steps = args.max_timesteps or env.spec.timestep_limit
        print('max_steps=%d' % max_steps)

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
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

            if (i + 1) % 20 == 0:
                print('returns', returns)
                print('mean return', np.mean(returns))
                print('std of return', np.std(returns))

                if args.stats:
                    with open(os.path.join('expert_data', 'rollout_' + str(args.num_rollouts) + '.stats'), 'a') as f:
                        f.write(args.envname + '_' + str(args.num_rollouts) + '\n')
                        f.write('mean return %f\n' % np.mean(returns))
                        f.write('std of return %f\n' % np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        print('observations shape = %s' % str(np.array(observations).shape))
        print('actions shape = %s' % str(np.array(actions).shape))

        if args.output_rollout:
            with open(os.path.join('expert_data', args.envname + '_' + str(args.num_rollouts) + '.pkl'), 'wb') as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()
