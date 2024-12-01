import matplotlib.pyplot as plt
import numpy as np

def plot_arr(arr, name):
    steps = [x for x in range(199999)]
    arr_np = np.array(arr)
    arr_np = arr_np.reshape(10, 199999).astype(float)
    plt.figure(figsize=(10,5))
    plt.xlabel('Steps')
    plt.ylabel(name)
    plt.title(name + " Results averaged over 10 Iterations on Topology 0")
    avg_arr = np.mean(arr_np, axis=0)
    print(avg_arr.shape)
    plt.plot(steps, avg_arr)
    #for i in range(10):
    #    plt.plot(steps, arr_np[i, :])

    plt.savefig(name + "_0.jpg")
    plt.show()


steps = []
reward = []
avg_delay = []
avg_e2e = []
avg_loss = []


with open("topology_0_report.txt", 'r') as f:
    for line in f:
        pairs = line.strip().split(',')
        for pair in pairs[1:]:
            key, value = pair.split(':')
            if key == ' Reward':
                reward.append(value)
            elif key == ' Avg Delay':
                avg_delay.append(value)
            elif key == ' Avg E2E Delay':
                avg_e2e.append(value)
            elif key == ' Avg Loss':
                avg_loss.append(value)


plot_arr(reward, 'reward')
plot_arr(avg_delay, 'avg_delay')
plot_arr(avg_e2e, 'Avg E2E Delay')
plot_arr(avg_loss, 'Avg Loss')
