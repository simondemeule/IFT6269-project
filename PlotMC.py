import matplotlib.pyplot as plt
import json

suffixe='new_update'


fig = plt.figure()
plt.ioff()
#If there is no file with a specific loss, just skip it
file=json.load(open('Gaussian_IMGS/forward_kl/ArtificialMC.txt', 'r'))

x=arr=[i for i in range(5, 50000, 100)]

plt.plot(x, file['Estimated'], label='Estimated')
plt.plot(x, file['True'], label='Real')
plt.xlabel('Sampling Size')
plt.ylabel('KL divergence')
plt.legend()
plt.title('Error induced by the MC sampling algorithm in calculating the f-divergence')
plt.savefig(f"Gaussian_IMGS/ArtificalMC.png")
plt.close(fig)