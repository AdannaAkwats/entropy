from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
from shannon import *
from generate_random_quantum import *
from evolution import *
from scipy.signal import savgol_filter


# AVERAGE PLOT OF AVG. DIFF USING DIFFERENT METHODS OF GENERATION, WITH
# ZHANG-YEUNG INEQUALITY, RESULTS IN y GOTTEN FRON print_tests.py
y = [4.66, 3.66, 0.315, 0.107, 0]
x = ["m2", "m1", "pure", "m3", "GHZ"]
yhat = savgol_filter(y, 5, 2)
plt.xlabel('Methods of generation')
plt.ylabel('Average difference of Zhang-Yeung inequality')
plt.plot(x, yhat, color='purple', linewidth = 2)
plt.show()


# FOR PLOTTING AVERGAGE ENTROPY DIFFERENCE BIT FLIP CHANNEL
def plot_channel_ent_diff(channel_func, channel_name):
    diff_e = [[] for i in range(11)]
    for n in range(1000):
        p = generate(2)
        prob = 0

        for i in range(11):
            e, ev = CTPT_entropy(channel_func,p, prob)
            diff_e[i].append(ev-e)
            prob = prob + 0.1

    d = []
    # avg. diff_e
    for i in range(11):
        s = sum(diff_e[i]) / 1000
        d.append(s)

    prob_list = [0.1*i for i in range(0,11)]
    plt.plot(prob_list, d)
    plt.xlabel('Probability')
    plt.ylabel(channel_name + ' channel entropy difference')
    plt.show()


plot_channel_ent_diff(bit_flip_channel, "Bit flip")
plot_channel_ent_diff(phase_flip_channel, "Phase flip")
plot_channel_ent_diff(bit_phase_flip_channel, "Bit phase flip")
plot_channel_ent_diff(depolarising_channel, "Depolarising")
