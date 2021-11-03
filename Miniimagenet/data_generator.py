import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class MiniImagenet(Dataset):

    def __init__(self, args, mode, tau=0.7, alpha=3):
        super(MiniImagenet, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.noise = args.noise
        self.mode = mode
        self.alpha = alpha
        self.tau = tau
        self.threshold = self.noise
        self.noise = 0.8

        if mode == 'train':
            if self.noise > 0:
                print("noise:", self.noise)
                self.noise_matrix = np.diag([1 - self.noise] * self.nb_classes)
                for i in range(self.nb_classes):
                    for j in range(self.nb_classes):
                        if j == i:
                            continue
                        self.noise_matrix[i][j] = self.noise / (self.nb_classes - 1)
                else:
                    raise NotImplementedError
                print(self.noise_matrix)
        else:
            self.noise = 0


        if mode == 'train' or mode == 'val':
            self.data_file = '{}/miniImagenet/mini_imagenet_train.pkl'.format(args.datadir)
            self.data = pickle.load(open(self.data_file, 'rb'))

            if mode == 'val':
                self.data_file = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
                self.data = pickle.load(open(self.data_file, 'rb'))

        elif mode == 'test':
            self.data_file = '{}/miniImagenet/mini_imagenet_test.pkl'.format(args.datadir)
            self.data = pickle.load(open(self.data_file, 'rb'))

        self.data = torch.tensor(np.transpose(self.data / np.float32(255), (0, 1, 4, 2, 3)))

        if (mode == 'train' and args.limit_data):
            if args.limit_classes == 16:
                chosen_classes_idx = [24, 63, 43, 34, 23, 50, 42, 19, 30, 29, 54, 35, 0, 21, 26, 45]
            elif args.limit_classes == 32:
                chosen_classes_idx = [36, 62, 54,  5,  9, 41,  1,  6,  2,  0, 27, 55, 12, 22, 15,  3, 34,
                                      49, 59, 11, 16, 35, 32, 18, 17, 43, 21, 42, 28, 60, 61, 37]
            elif args.limit_classes == 48:
                chosen_classes_idx = [ 7, 22, 61, 18, 14, 30, 46,  4, 32,  6, 15, 48,  5, 25, 41, 54, 42,
                                       60, 58, 29, 53, 27, 50, 55, 19, 45, 52,  9, 44, 13, 28, 63, 62, 57,
                                       56, 33, 20, 47,  3, 12, 39, 17, 51, 49, 31, 24, 34, 43]
            elif args.limit_classes == 64:
                chosen_classes_idx = np.arange(64)
            else:
                raise NotImplementedError
            self.data = self.data[chosen_classes_idx]

        self.classes_idx = np.arange(self.data.shape[0])
        self.samples_idx = np.arange(self.data.shape[1])

    def __len__(self):
        return self.args.metatrain_iterations*self.args.meta_batch_size

    def __getitem__(self, index):

        support_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.query_size, 3, 84, 84)))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        noisy_or_not = torch.zeros(self.args.meta_batch_size)

        for meta_batch_id in range(self.args.meta_batch_size):

            self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)

            noise = self.noise
            if np.random.random() > self.threshold:
                noise = 0

            if not (noise > 0 and self.mode == 'train'):
                for j in range(self.nb_classes):
                    np.random.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                :self.k_shot], ...]
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                        self.choose_classes[
                            j], choose_samples[
                                self.k_shot:], ...]
                    support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

            else:
                noisy_or_not[meta_batch_id] = 1
                x = torch.zeros((self.set_size + self.query_size, 3, 84, 84))
                y = torch.zeros(self.set_size + self.query_size)
                # randomly sample 80 pictures
                for j in range(self.nb_classes):
                    np.random.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    x[j * (self.k_shot + self.k_query) : (j+1) * (self.k_shot + self.k_query)] \
                        = self.data[self.choose_classes[j], choose_samples, ...]
                    y[j * (self.k_shot + self.k_query) : (j+1) * (self.k_shot + self.k_query)] \
                        = j

                noisy_y = y.clone()
                for i in range(len(y)):
                    noisy_y[i] = np.random.choice(self.nb_classes, p=self.noise_matrix[int(y[i])])

                support_idxes = []
                for j in range(self.nb_classes):
                    idx = np.where(noisy_y == j)[0]
                    np.random.shuffle(idx)
                    idx = idx[:self.k_shot]
                    support_idxes.append(idx)

                support_idxes = np.concatenate(support_idxes)
                query_idxes = np.setdiff1d(np.arange(len(x)), support_idxes)
                support_idxes = np.concatenate([support_idxes, query_idxes[self.query_size:]])
                query_idxes = query_idxes[:self.query_size]

                support_x[meta_batch_id] = x[support_idxes]
                support_y[meta_batch_id] = noisy_y[support_idxes]
                query_x[meta_batch_id] = x[query_idxes]
                query_y[meta_batch_id] = y[query_idxes]

            if not self.args.mix:
                support_sample = np.arange(self.set_size)
                query_sample = np.arange(self.query_size)
                np.random.shuffle(support_sample)
                np.random.shuffle(query_sample)

                support_x[meta_batch_id] = support_x[meta_batch_id][support_sample]
                support_y[meta_batch_id] = support_y[meta_batch_id][support_sample]
                query_x[meta_batch_id] = query_x[meta_batch_id][query_sample]
                query_y[meta_batch_id] = query_y[meta_batch_id][query_sample]
        if self.mode == 'train':
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), noisy_or_not
        else:
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)


class MiniImagenet_MW(Dataset):

    def __init__(self, args, mode, diverse=True):
        super(MiniImagenet_MW, self).__init__()
        self.args = args
        self.noise = args.noise
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.train_train_count = 0
        self.train_val_count = 0
        self.diverse = diverse
        self.mode = mode
        self.threshold = self.noise
        self.noise = 0.8

        if mode == 'train':
            self.data_file = '{}/miniImagenet/mini_imagenet_train.pkl'.format(args.datadir)
            self.data_file_val = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
            if self.noise > 0:
                print("noise:", self.noise)
                self.noise_matrix = np.diag([1 - self.noise] * self.nb_classes)
                for i in range(self.nb_classes):
                    for j in range(self.nb_classes):
                        if j == i:
                            continue
                        self.noise_matrix[i][j] = self.noise / (self.nb_classes - 1)
                print(self.noise_matrix)
        elif mode == 'val':
            self.data_file = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
            self.noise = 0
        elif mode == 'test':
            self.data_file = '{}/miniImagenet/mini_imagenet_test.pkl'.format(args.datadir)
            self.noise = 0

        self.data = pickle.load(open(self.data_file, 'rb'))


        if mode == 'train' and args.limit_data:
            if args.limit_classes == 16:
                chosen_classes_idx = [24, 63, 43, 34, 23, 50, 42, 19, 30, 29, 54, 35, 0, 21, 26, 45]
            elif args.limit_classes == 32:
                chosen_classes_idx = [36, 62, 54, 5, 9, 41, 1, 6, 2, 0, 27, 55, 12, 22, 15, 3, 34,
                                      49, 59, 11, 16, 35, 32, 18, 17, 43, 21, 42, 28, 60, 61, 37]
            elif args.limit_classes == 48:
                chosen_classes_idx = [7, 22, 61, 18, 14, 30, 46, 4, 32, 6, 15, 48, 5, 25, 41, 54, 42,
                                      60, 58, 29, 53, 27, 50, 55, 19, 45, 52, 9, 44, 13, 28, 63, 62, 57,
                                      56, 33, 20, 47, 3, 12, 39, 17, 51, 49, 31, 24, 34, 43]
            elif args.limit_classes == 64:
                chosen_classes_idx = np.arange(64)
            else:
                raise NotImplementedError
            self.data = self.data[chosen_classes_idx]

        if mode =='train':
            self.data_val = pickle.load(open(self.data_file_val, 'rb'))
            self.data_val = torch.tensor(np.transpose(self.data_val / np.float32(255), (0, 1, 4, 2, 3)))
            self.classes_idx_val = np.arange(self.data_val.shape[0])

        self.data = torch.tensor(np.transpose(self.data / np.float32(255), (0, 1, 4, 2, 3)))
        self.classes_idx = np.arange(self.data.shape[0])
        self.samples_idx = np.arange(self.data.shape[1])

    def get_sampled_data(self, data, index, setting):
        if setting == 'train_train' and self.args.sampling_method == 'ATS':
            task_num = self.args.buffer_size
        else:
            task_num = self.args.meta_batch_size

        support_x = torch.FloatTensor(torch.zeros((task_num, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros((task_num, self.query_size, 3, 84, 84)))

        support_y = np.zeros([task_num, self.set_size])
        query_y = np.zeros([task_num, self.query_size])

        noisy_or_not = torch.zeros(task_num)

        for meta_batch_id in range(task_num):
            if self.mode == 'train':
                if setting == 'train_train':
                    if not self.diverse or (self.train_train_count + 1) * self.nb_classes > len(self.classes_idx):
                        self.train_train_count = 0
                        np.random.shuffle(self.classes_idx)
                    self.choose_classes = self.classes_idx[self.train_train_count * self.nb_classes:
                                                           (self.train_train_count + 1) * self.nb_classes]
                    self.train_train_count += 1
                elif setting == 'train_val':
                    if not self.diverse or (self.train_val_count + 1) * self.nb_classes > len(self.classes_idx_val):
                        self.train_val_count = 0
                        np.random.shuffle(self.classes_idx_val)
                    self.choose_classes = self.classes_idx_val[self.train_val_count * self.nb_classes:
                                                               (self.train_val_count + 1) * self.nb_classes]
                    self.train_val_count += 1
            else:
                self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)

            noise = self.noise
            if np.random.random() > self.threshold:
                noise = 0

            if noise > 0: noisy_or_not[meta_batch_id] = 1

            if not (noise > 0 and setting == 'train_train'):
                for j in range(self.nb_classes):
                    np.random.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = data[
                        self.choose_classes[
                            j], choose_samples[
                                :self.k_shot], ...]
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = data[
                        self.choose_classes[
                            j], choose_samples[
                                self.k_shot:], ...]
                    support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

            else:
                x = torch.zeros((self.set_size + self.query_size, 3, 84, 84))
                y = torch.zeros(self.set_size + self.query_size)
                # random sample 80 pictures
                for j in range(self.nb_classes):
                    np.random.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    x[j * (self.k_shot + self.k_query) : (j+1) * (self.k_shot + self.k_query)] \
                        = data[self.choose_classes[j], choose_samples, ...]
                    y[j * (self.k_shot + self.k_query) : (j+1) * (self.k_shot + self.k_query)] \
                        = j

                noisy_y = y.clone()
                for i in range(len(y)):
                    noisy_y[i] = np.random.choice(self.nb_classes, p=self.noise_matrix[int(y[i])])

                support_idxes = []
                for j in range(self.nb_classes):
                    idx = np.where(noisy_y == j)[0]
                    np.random.shuffle(idx)
                    idx = idx[:self.k_shot]
                    support_idxes.append(idx)

                support_idxes = np.concatenate(support_idxes)
                query_idxes = np.setdiff1d(np.arange(len(x)), support_idxes)
                support_idxes = np.concatenate([support_idxes, query_idxes[self.query_size:]])
                query_idxes = query_idxes[:self.query_size]

                support_x[meta_batch_id] = x[support_idxes]
                support_y[meta_batch_id] = y[support_idxes]
                query_x[meta_batch_id] = x[query_idxes]
                query_y[meta_batch_id] = noisy_y[query_idxes]

            support_sample = np.arange(self.set_size)
            query_sample = np.arange(self.query_size)
            np.random.shuffle(support_sample)
            np.random.shuffle(query_sample)

            support_x[meta_batch_id] = support_x[meta_batch_id][support_sample]
            support_y[meta_batch_id] = support_y[meta_batch_id][support_sample]
            query_x[meta_batch_id] = query_x[meta_batch_id][query_sample]
            query_y[meta_batch_id] = query_y[meta_batch_id][query_sample]
        if setting == 'train_train':
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y), noisy_or_not
        else:
            return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        if self.mode == 'train':
            support_x, support_y, query_x, query_y, noisy_or_not = self.get_sampled_data(self.data, index, setting='train_train')
            support_x_val, support_y_val, query_x_val, query_y_val = self.get_sampled_data(self.data, index, setting='train_val')

            return support_x, support_y, query_x, query_y, support_x_val, support_y_val, query_x_val, query_y_val, noisy_or_not
        else:
            support_x, support_y, query_x, query_y = self.get_sampled_data(self.data, index, setting='test')

            return support_x, support_y, query_x, query_y
