from sklearn import manifold
import scipy
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.cbook import get_sample_data
game = 'breakout'
run_dir = 'embeddings/breakoutA'



def imscatter(x, y, images, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    # try:
    #     image = plt.imread(image)
    # except TypeError:
    #     # Likely already an array...
    #     pass
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0,im_num in zip(x, y,images):
        image_path = '{0}/{1}{2}.png'.format(run_dir,game,im_num)
        image = scipy.misc.imread(image_path,mode='L')
        # scipy.misc.imshow(image)
        im = OffsetImage(image,cmap='gray', zoom=zoom)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=True)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


if __name__ == '__main__':
    num_samples = 100
    num_images = 200
    embeddings = np.load('{}/{}.npy'.format(run_dir,game))
    # print embeddings.shape

    if os.path.exists('{}/reduced_{}.p'.format(run_dir,game)):
        low_dim_embs = pickle.load(open('{}/reduced_{}.p'.format(run_dir,game),'r'))
    else:
        tsne = manifold.TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000,verbose=1)
        low_dim_embs = tsne.fit_transform(embeddings)
        pickle.dump(low_dim_embs,open('{}/reduced_{}.p'.format(run_dir,game),'w'))

    # points = np.random.randint(0,embeddings.shape[0],size=num_samples)

    # sampled = low_dim_embs[points]
    images = np.random.randint(0,embeddings.shape[0],num_images)
    # images = points[image_idx]
    image_pts = low_dim_embs[images]
    images = images+1
    # print images
    # print image_pts
    fig, ax = plt.subplots()
    # ax.scatter(sampled[:,0],sampled[:,1])
    imscatter(image_pts[:,0],image_pts[:,1],images,ax=ax,zoom=2)

    plt.show()
