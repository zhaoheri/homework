#!/usr/bin/env python
import matplotlib.pyplot as plt


def plot_2_3():
    epochs = [1, 3, 5, 10, 20, 30]
    rewards = [4441.38, 4447.10, 4707.91, 4555.75, 4817.26, 4796.71]
    stds = [210.81, 1021.70, 337.22, 933.02, 107.66, 105.68]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, rewards)
    plt.errorbar(epochs, rewards, yerr=stds, fmt='o')
    plt.suptitle('Behavioral Cloning: Epochs vs. Reward', fontsize=15)
    plt.xlabel('Number of Epochs')
    plt.ylabel('Mean Reward')
    plt.xlim([0, 35])
    plt.legend(["Mean", "Std"])
    plt.savefig("2-3.png", dpi=300)


def plot_3_2():

    interation = [1, 2, 3, 4, 5, 7, 10]
    mean = [-6.59, -4.39, -3.66, -3.39, -4.51, -4.38, -3.06]
    std = [2.91, 1.88, 1.63, 1.19, 2.15, 1.83, 1.41]

    plt.figure(figsize=(6, 4))
    plt.plot(interation, mean, label="DAgger Mean")
    plt.errorbar(interation, mean, yerr=std, fmt='o', label="DAgger Std")
    plt.suptitle('DAgger Iterations vs. Rewards', fontsize=15)

    plt.xlabel('DAgger Iteration')
    plt.ylabel('Mean Reward')
    plt.xlim([0, 11])
    plt.ylim([-10, -2])

    plt.axhline(y=-3.37, color='orange', label='Expert')
    plt.axhline(y=-5.21, color='r', label='Behavioral Cloning')
    plt.legend()
    plt.savefig("3-2.png", dpi=300)


plot_2_3()
plot_3_2()