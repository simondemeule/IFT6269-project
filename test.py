
import json

with open(f"MNIST_IMGS/total_variation/DivergenceOther.txt", "r") as file:
    other = json.load(file)

for item in other:
    print(item['divergence'])