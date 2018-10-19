import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import pickle


def part1_q1():
    data = pd.DataFrame((pickle.load(open('data/PongNoFrameskip-v4/08-10-2018_06-18-13_lr_vanilla.pkl', 'rb'))))

    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(x='timestep', y='mean_reward', data=data, label="mean_reward")
    sns.lineplot(x='timestep', y='best_reward', data=data, label='best_reward')
    plt.xlabel("Timestep")
    plt.ylabel("Rewards")
    plt.legend(loc='best').draggable()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()


def part1_q2():
    data_vanilla = pd.DataFrame((pickle.load(open('data/PongNoFrameskip-v4/08-10-2018_06-18-13_lr_vanilla.pkl', 'rb'))))
    data_dq = pd.DataFrame((pickle.load(open('data/PongNoFrameskip-v4/07-10-2018_04-57-19_dq.pkl', 'rb'))))

    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(x='timestep', y='mean_reward', data=data_vanilla, label="vanilla mean_reward")
    sns.lineplot(x='timestep', y='best_reward', data=data_vanilla, label='vanilla best_reward')
    sns.lineplot(x='timestep', y='mean_reward', data=data_dq, label="double-q mean_reward")
    sns.lineplot(x='timestep', y='best_reward', data=data_dq, label='double-q best_reward')
    plt.xlabel("Timestep")
    plt.ylabel("Rewards")
    plt.xlim(0, 4.3e6)
    plt.legend(loc='best').draggable()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()


def part1_q3():
    data_rl1 = pd.DataFrame(pickle.load(
        open('data/PongNoFrameskip-v4/07-10-2018_04-57-19_dq.pkl', 'rb')))
    data_rl01 = pd.DataFrame(pickle.load(
        open('data/PongNoFrameskip-v4/07-10-2018_15-56-26_lr_0.1.pkl', 'rb')))
    data_rl05 = pd.DataFrame(pickle.load(
            open('data/PongNoFrameskip-v4/08-10-2018_00-56-06_lr_0.5.pkl', 'rb')))
    data_rl10 = pd.DataFrame(pickle.load(
            open('data/PongNoFrameskip-v4/07-10-2018_21-09-03_lr_10.0.pkl', 'rb')))

    sns.set(style="darkgrid", font_scale=1.5)
    sns.lineplot(x='timestep', y='mean_reward', data=data_rl1, label='lr=1 mean_reward')
    sns.lineplot(x='timestep', y='best_reward', data=data_rl1, label='lr=1 best_reward')
    sns.lineplot(x='timestep', y='mean_reward', data=data_rl01, label='lr=0.1 mean_reward')
    sns.lineplot(x='timestep', y='best_reward', data=data_rl01, label='lr=0.1 best_reward')
    sns.lineplot(x='timestep', y='mean_reward', data=data_rl05, label='lr=0.5 mean_reward')
    sns.lineplot(x='timestep', y='best_reward', data=data_rl05, label='lr=0.5 best_reward')
    sns.lineplot(x='timestep', y='mean_reward', data=data_rl10, label='lr=10 mean_reward')
    sns.lineplot(x='timestep', y='best_reward', data=data_rl10, label='lr=10 best_reward')
    plt.xlabel("Timestep")
    plt.ylabel("Rewards")
    plt.xlim(0, 4e6)
    plt.legend(loc='best').draggable()
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.show()

# part1_q1()
# part1_q2()
part1_q3()
