import matplotlib.pyplot as plt

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
            if key == 'Reward':
                reward.append(value)
            elif key == 'Avg Delay':
                avg_delay.append(value)
            elif key == 'Avg E2E Delay':
                avg_e2e.append(value)
            elif key == 'Avg Loss':
                avg_loss.append(value)

steps = [x for x in range(len(reward))]

plt.figure(figsize=(10, 5))
plt.plot(reward, label='Reward', marker='o')
plt.plot(avg_delay, label='Delay', marker='s')
plt.plot(avg_e2e, label='End 2 End Delay')
plt.plot(avg_loss, label='Loss')
plt.xlabel("Step")
plt.ylabel('Values')
plt.savefig('report.jpg')
plt.show()

