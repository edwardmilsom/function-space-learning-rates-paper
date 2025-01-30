import argparse
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument('min', type=float)
argparser.add_argument('max', type=float)
argparser.add_argument('--steps', type=int, default=10)

args = argparser.parse_args()

min = args.min
max = args.max

exp_loglinspace = np.exp(np.linspace(np.log(min), np.log(max), args.steps))

for lr in exp_loglinspace:
    print(str(lr) + " ", end="")
print()