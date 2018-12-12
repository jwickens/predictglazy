from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt
from datasets import GlazeFlattenedImageDataset
import json
import time

ds = GlazeFlattenedImageDataset()

start_time = time.time()
som = MiniSom(x=50, y=50, input_len=len(ds[0]), sigma=0.5, learning_rate=0.4)
som.random_weights_init(ds)
som.train_random(data=ds, num_iteration=1000)

train_time = time.time()
print('trained in %i seconds' % (train_time - start_time))

counts = {}
winners = []

for i, x in enumerate(ds):
    w = som.winner(x)
    winners.append([int(w[0]), int(w[1])])
    key = '%i,%i' % w
    if key in counts:
        counts[key] += 1
    else:
        counts[key] = 1

print('evaluated in %i seconds' % (time.time() - train_time))

with open('som_winners.json', 'w') as f:
    json.dump({'counts': counts, 'winners': winners}, f)

print('done')
