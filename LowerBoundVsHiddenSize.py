import matplotlib.pyplot as plt
import json



fig = plt.figure()
plt.ioff()
#If there is no file with a specific loss, just skip it
file=json.load(open('Gaussian_IMGS/forward_kl/LowerBoundVsCapacity.txt', 'r'))

x=file['crit_size']

plt.plot(x, file['Estimated'], label='Estimated')
plt.plot(x, file['True'], label='Real')
plt.plot(x, file['Sampled'], label='Sampled')
plt.xlabel('Number of hidden neurones per layer')
plt.ylabel('KL divergence')
plt.legend()
plt.title('Real, estimated, and sampled \nKL divergence vs hidden size')
plt.savefig(f"Gaussian_IMGS/LowerBoundVsHiddenSize.png")
plt.close(fig)