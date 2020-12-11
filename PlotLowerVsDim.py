import matplotlib.pyplot as plt
import json



fig = plt.figure()
plt.ioff()
#If there is no file with a specific loss, just skip it
file=json.load(open('Gaussian_IMGS/forward_kl/LowerBoundVsDim.txt', 'r'))

x=file['num_dim']

plt.plot(x, file['Estimated'], label='Estimated')
plt.plot(x, file['True'], label='Real')
plt.plot(x, file['Sampled'], label='Sampled')
plt.xlabel('number of dimension')
plt.ylabel('KL divergence')
plt.legend()
plt.title('Real, estimated, and sampled \nKL divergence vs number of dimension')
plt.savefig(f"Gaussian_IMGS/LowerBoundVsDim.png")
plt.close(fig)