import matplotlib.pyplot as plt
import json



fig = plt.figure()
plt.ioff()
#If there is no file with a specific loss, just skip it
file=json.load(open('Gaussian_IMGS/forward_kl/LowerBoundVsShiftingDist.txt', 'r'))

x=file['Shift']

plt.plot(x, file['Estimated'], label='Estimated')
plt.plot(x, file['True'], label='Real')
plt.plot(x, file['Sampled'], label='Sampled')
plt.xlabel('Step Size')
plt.ylabel('KL divergence')
plt.legend()
plt.ylim([-100, 100])
plt.title('Real, estimated, and sampled \nKL divergence vs Shifting Distribution')
plt.savefig(f"Gaussian_IMGS/LowerBoundVsShift.png")
plt.close(fig)