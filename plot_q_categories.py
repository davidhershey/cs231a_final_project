from sklearn import manifold
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import random
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
from collections import defaultdict
game = 'pong'
run_dir = 'embeddings/q'

def filter(q,eps):
    idxs = q.argsort()[-2:][::-1]
    if (q[idxs[0]]-q[idxs[1]]) > eps:
        return True
    return False
    # exit(0)


if __name__ == '__main__':
    qs = np.load('{}/{}.npy'.format(run_dir,game))
    num_acts = qs.shape[1]

    cats = defaultdict(list)

    for i in range(qs.shape[0]):
        q = qs[i]
        if filter(q,.0001):
            cats[np.argmax(q)].append(i)

    for i in range(num_acts):
        print i, len(cats[i])

        num_samps = len(cats[i])
        if len(cats[i]) < num_samps or len(cats[i]) == 0:
            continue
        samps = random.sample(cats[i],num_samps)
        print samps
        image = scipy.misc.imread(run_dir + '/' + game + '{}.png'.format(samps[0]+1),mode='L')
        for s in range(1,num_samps):
            im_num = s+1
            image += (scipy.misc.imread(run_dir + '/' + game + '{}.png'.format(im_num),mode='L'))
        image /= num_samps
        plt.imshow(image,cmap='gray')
        plt.show()
    # print embeddings.shape
