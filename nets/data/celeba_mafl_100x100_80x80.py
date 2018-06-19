import os
import cv2
import numpy as np
import threading
import random
from multiprocessing.dummy import Pool
from multiprocessing import cpu_count


class Net:
    def __init__(self, subset_name='train', options = None):

        """ Module for loading CelebA data
        :param subset_name: "train", "validation", "test"
        :param cache_dir: (default) "var/data/OMNIGLOT"
        """
        if subset_name == 'train':
            self._mode = '0'
        elif subset_name == 'validation':
            self._mode = '1'
        elif subset_name == 'test':
            self._mode = '2'
        else:
            self._mode = '0'

        self._subset_name = subset_name
        self._debug = False
        self._shuffle = False
        self._cache_size = 3000
        self._mean_reduce = False
        self._mean = [5.0, 10.0, 15.0]
        if options != None and options != {}:
            if 'cache_size' in options:
                self._cache_size = options['cache_size']
            if 'mean_reduce' in options:
                self._mean_reduce = options['mean_reduce']
            if 'shuffle' in options:
                self._shuffle = options['shuffle']
            if 'debug' in options:
                self._debug = options['debug']

        current_path = os.path.dirname(os.path.abspath(__file__))
        root_path = current_path[:-9]
        self._imname = root_path+'data/celeba_images/Eval/list_eval_partition.txt'
        self._maflname_train = root_path+'data/celeba_data/training.txt'
        self._maflname_test = root_path+'data/celeba_data/testing.txt'
        self._impath = root_path+'data/celeba_images/Img/img_align_celeba_png/'
        with open(self._imname, 'r') as f:
            self._lines = f.read().splitlines()
        self._imlist = [line.split(' ')[0] for line in self._lines if line.split(' ')[1] == self._mode]
        self._celeba_test = [line.split(' ')[0] for line in self._lines if line.split(' ')[1] == '2']
        with open(self._maflname_train, 'r') as f:
            self._lines = f.read().splitlines()
        self._mafllist_train = [line.split(' ')[0] for line in self._lines]
        with open(self._maflname_test, 'r') as f:
            self._lines = f.read().splitlines()
        self._mafllist_test = [line.split(' ')[0] for line in self._lines]
        if subset_name == 'train':
            self._imlist = sorted(list(set(self._imlist)-set(self._mafllist_test)))
        if subset_name == 'mafl_train':
            self._imlist = self._mafllist_train
        if subset_name == 'test':
            self._imlist = self._mafllist_test
        if subset_name == 'celeba_test':
            self._imlist = self._celeba_test
        if subset_name == 'demo':
            import glob
            self._imlist = sorted(glob.glob(root_path+'demo/input/*.jpg'))

        if "chosen_indexes" in options and options["chosen_indexes"] is not None:
            chosen_index = options["chosen_indexes"]
            self._imlist = list(self._imlist[i] for i in chosen_index)

        self._num_samples = len(self._imlist)
        self._waitlist = list(range(len(self._imlist)))
        if self._shuffle:
            random.shuffle(self._waitlist)
        self._dataset = None
        self._cur_pos = 0  # num of sample done in this epoch
        self._cur_epoch = 0  # current num of epoch
        self._cur_iter = 0  # num of batches returned
        self._num_fields = 1  # number of fields need to return (image, label)
        self._out_h = 80
        self._out_w = 80

        self._image_cache = []

        self._lock = threading.Lock()

        self._pool_size = cpu_count()

        self._pool = Pool(self._pool_size)
        self._cache_thread = threading.Thread(target=self.preload_dataset)
        self._cache_thread.start()

    def read_image(self, i):
        if self._subset_name == 'demo':
            image_name = self._imlist[i]
            image_arr = cv2.imread(image_name)
            image_arr = cv2.resize(image_arr, (80, 80))
            result = image_arr.astype(np.float32) / np.array(255., dtype=np.float32)
        else:
            image_name = self._impath + self._imlist[i].split('.')[0] + '.png'
            # The channel for cv2.imread is B, G, R
            image_arr = cv2.imread(image_name)
            image_arr = cv2.resize(image_arr, (100, 100))
            height, width, channels = image_arr.shape
            margin_h = int(np.round((height - self._out_h) / 2))
            margin_w = int(np.round((width - self._out_w) / 2))
            cropped_im = image_arr[margin_h:self._out_h + margin_h, margin_w:self._out_w + margin_w, :]
            result = cropped_im.astype(np.float32) / np.array(255., dtype=np.float32)
        result[:, :, [0, 1, 2]] = result[:, :, [2, 1, 0]]
        return result

    def __call__(self, *args, **kwargs):
        return self.next_batch(*args, **kwargs)

    def num_samples(self):
        return self._num_samples

    def epoch(self):
        return self._cur_epoch

    def iter(self):
        return self._cur_iter

    def num_fields(self):
        return self._num_fields

    def num_samples_finished(self):
        return self._cur_pos

    def reset(self):
        """ Reset the state of the data loader
        E.g., the reader points at the beginning of the dataset again
        :return: None
        """
        self._cur_pos = 0
        self._cur_epoch = 0
        self._cur_iter = 0
        self._waitlist = list(range(len(self._imlist)))
        if self._shuffle:
            random.shuffle(self._waitlist)
        tmp = 0
        while self._cache_thread.isAlive():
            tmp+=1
        self._cache_thread = threading.Thread(target=self.preload_dataset)
        self._lock.acquire()
        self._image_cache = []
        self._lock.release()
        self._cache_thread.start()

    def preload_dataset(self):
        if self._debug:
            print("preload")
        if len(self._image_cache) > self._cache_size:
            return
        else:
            while len(self._image_cache) < 1000:
                if len(self._waitlist) < 1000:
                    self._waitlist += list(range(len(self._imlist)))
                    if self._shuffle:
                        random.shuffle(self._waitlist)

                results = self._pool.map(self.read_image, self._waitlist[:1000])
                del self._waitlist[:1000]
                self._lock.acquire()
                self._image_cache = self._image_cache + results
                self._lock.release()
            if self._debug:
                print(len(self._image_cache))

    def next_batch(self, batch_size):
        """ fetch the next batch
        :param batch_size: next batch_size
        :return: a tuple includes all data
        """
        if batch_size < 0:
            batch_size = 0
        if self._cache_size < 3 * batch_size:
            self._cache_size = 3 * batch_size

        this_batch = [None] * self._num_fields

        if len(self._image_cache) < batch_size:
            if self._debug:
                print("Blocking!!, Should only appear once with proper setting")

            if not self._cache_thread.isAlive():
                self._cache_thread = threading.Thread(target=self.preload_dataset)
                self._cache_thread.start()
            self._cache_thread.join()

            self._lock.acquire()
            this_batch[0] = self._image_cache[0:batch_size]
            del self._image_cache[0:batch_size]
            self._lock.release()
        else:
            self._lock.acquire()
            this_batch[0] = self._image_cache[0:batch_size]
            del self._image_cache[0:batch_size]
            self._lock.release()
            if not self._cache_thread.isAlive():
                self._cache_thread = threading.Thread(target=self.preload_dataset)
                self._cache_thread.start()

        self._cur_iter += 1
        self._cur_pos = self._cur_pos + batch_size
        if self._cur_pos >= self._num_samples:
            self._cur_epoch += 1
            self._cur_pos = self._cur_pos % self._num_samples

        return this_batch

    @staticmethod
    def output_types():  # only used for net instance
        t = ["float32"]
        return t

    @staticmethod
    def output_shapes():
        t = [(None, 80, 80, 3)]  # None for batch size
        return t

    @staticmethod
    def output_ranges():
        return [1.]

    @staticmethod
    def output_keys():
        return ["data"]

if __name__ == '__main__':
    main()



