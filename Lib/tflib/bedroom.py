from __future__ import print_function
import numpy as np
import scipy.misc
import time

def make_generator(path, n_files, batch_size):
    epoch_count = [1]
    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        #files = np.arange(n_files)
        files = listfile(path)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            #image = scipy.misc.imread("{}/{}.jpg".format(path, str(i+1).zfill(7)))
            image = scipy.misc.imread(i)
            images[n % batch_size] = image.transpose(2,0,1)
            if n > 0 and n % batch_size == 0:
                yield (images,)
    return get_epoch

def load(batch_size, data_dir='~/kjliu/program-file/BEGAN-tensorflow/data/Bedrrom/splits'):
    return (
        make_generator(data_dir+'/train', 300000, batch_size),
        make_generator(data_dir+'/valid', 10000, batch_size)
    )

if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print("{}\t{}".format(str(time.time() - t0), batch[0][0,0,0,0]))
        if i == 1000:
            break
        t0 = time.time()


from os.path import join, isfile
from os import listdir
def listfile(path, onlyfiles = False):
    filenames = [join(path, f) for f in listdir(path) if isfile(join(path, f))]
    return filenames
