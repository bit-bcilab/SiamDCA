

import numpy as np

from configs.DataPath import TRAIN_PATH, ROOT_PATH, DET_PATH, TRAIN_JSON_PATH
from utils.rand import random_sys

import cv2
import json
import random


class DataLoader(object):
    def __init__(self, data_settings, read_all_boxes=False):
        self.dataset_trained = []
        self.data_num = 0
        self.num_train = 0
        self.num_val = 0
        self.sub_datasets = {}
        self.val_index = []
        self.read_all_boxes = read_all_boxes

        for sub_dataset in data_settings['dataset_used']:

            self.sub_datasets[sub_dataset] = data_settings[sub_dataset]

            with open(TRAIN_JSON_PATH + self.sub_datasets[sub_dataset]['label_path']) as f:
                data = json.load(f)
                f.close()
            self.sub_datasets[sub_dataset]['data'] = data

            num_data = self.sub_datasets[sub_dataset]['num_data']
            assert num_data == len(data)

            multiply = self.sub_datasets[sub_dataset]['multiply']
            num_train = self.sub_datasets[sub_dataset]['num_train']
            num_val = self.sub_datasets[sub_dataset]['num_val']
            num_train_objects = self.sub_datasets[sub_dataset]['num_train_objects']
            num_val_objects = self.sub_datasets[sub_dataset]['num_val_objects']
            assert num_train_objects <= num_train * multiply
            assert num_val_objects <= num_val * multiply

            dataset = [sub_dataset] * num_data
            keys = list(data.keys())
            index = list(zip(dataset, keys))
            random.shuffle(index)

            if num_val > 0:
                train_index = index[:-num_val]
                val_index = index[-num_val:]

                val_index = val_index * multiply
                random.shuffle(val_index)
                val_index = val_index[:num_val_objects]
            else:
                train_index = index
                val_index = []

            self.sub_datasets[sub_dataset].update(dict(train_index=train_index, val_index=val_index))

            self.val_index += val_index
            self.num_train += num_train_objects
            self.num_val += num_val_objects

            print('load  ' + sub_dataset + ' done, train: %d, val: %d' % (num_train_objects, num_val_objects))
        print('Dataloader done. Total train number: %d, Total val number: %d' % (self.num_train, self.num_val))
        random.shuffle(self.val_index)
        self.build_train_index()

    def build_train_index(self):
        self.train_index = []
        for sub_dataset in self.sub_datasets:
            sub_index = self.sub_datasets[sub_dataset]['train_index'].copy()
            if sub_index:
                random.shuffle(sub_index)

                sub_index = sub_index[:self.sub_datasets[sub_dataset]['num_train']]
                sub_index *= self.sub_datasets[sub_dataset]['multiply']
                random.shuffle(sub_index)

                sub_index = sub_index[:self.sub_datasets[sub_dataset]['num_train_objects']]

                self.dataset_trained.append(sub_dataset)

            self.train_index += sub_index
        random.shuffle(self.train_index)

    def get_random_data(self, read_all_boxes=False):
        random_dataset = random.choice(self.dataset_trained)
        random_index = random.choice(self.sub_datasets[random_dataset]['train_index'])
        return self.get_data(random_index, read_pair=False, read_all_boxes=read_all_boxes)

    def read(self, idx, validate, positive):
        if validate:
            index = self.val_index[idx]
        else:
            index = self.train_index[idx]

        if positive:
            all_boxes, search_img, search_box, template_img, template_box = self.get_data(index, read_pair=True, read_all_boxes=self.read_all_boxes)
        else:
            all_boxes, search_img, search_box = self.get_data(index, read_pair=False, read_all_boxes=self.read_all_boxes)
            _, template_img, template_box = self.get_random_data(read_all_boxes=False)

        return all_boxes, search_img, search_box, template_img, template_box

    def get_data(self, index, read_pair=True, read_all_boxes=False):
        dataset = index[0]
        index = index[1]
        data = self.sub_datasets[dataset]['data'][index]
        match_range = self.sub_datasets[dataset]['match_range']
        path = TRAIN_PATH[dataset] + '/' + index
        all_boxes = []

        if dataset in ['DET', 'DET_val', 'COCO', 'COCO_val']:
            if dataset == 'DET' or dataset == 'DET_val':
                if index[0] == 'a':
                    search_path = ROOT_PATH + DET_PATH + index[:index.index('_')] + '/' + index[2:] + '.JPEG'
                else:
                    search_path = path + '.JPEG'
            else:
                search_path = path + '.jpg'

            samples = list(data.keys())
            num_sample = len(data)
            if num_sample > 1:
                search_index = random.randint(0, num_sample - 1)
            else:
                search_index = 0
            search_box = data[samples[search_index]]['000000']

            if read_pair:
                template_path = search_path

            if read_all_boxes:
                for i in range(num_sample):
                    if i != search_index:
                        all_boxes.append(np.array(data[samples[i]]['000000'], dtype=np.float32))

        elif dataset in ['VID', 'VID_val']:
            num_sample = len(data)
            samples = list(data.keys())
            if num_sample == 1:
                sample_index = 0
            else:
                sample_index = random.randint(0, num_sample - 1)

            sample_data = data[samples[sample_index]]
            frames = list(sample_data.keys())
            num_frame = len(frames)
            search_index = random.randint(0, num_frame - 1)
            search_frame = frames[search_index]
            search_path = path + '/' + search_frame + '.JPEG'
            search_box = sample_data[search_frame]

            if read_pair:
                if match_range == 'all':
                    template_index = random.randint(0, num_frame - 1)
                elif match_range == 'init':
                    template_index = 0
                elif match_range == 'mix':
                    if random_sys() > 0.5:
                        template_index = 0
                    else:
                        template_index = random.randint(0, num_frame - 1)
                else:
                    template_index = random.randint(max(search_index - match_range, 0),
                                                    min(search_index + match_range, num_frame) - 1)

                template_path = path + '/' + frames[template_index] + '.JPEG'
                template_box = sample_data[frames[template_index]]

            if read_all_boxes:
                if num_sample > 1:
                    for i in range(num_sample):
                        if i != sample_index:
                            sample_frames = list(data[samples[i]].keys())
                            if search_frame in sample_frames:
                                all_boxes.append(np.array(data[samples[i]][search_frame], dtype=np.float32))

        elif dataset in ['GOT', 'GOT_val', 'LaSOT']:
            if dataset == 'LaSOT':
                path = path + '/img/'

            frames = list(data.keys())
            num_frame = len(frames)
            search_index = random.randint(0, num_frame - 1)
            search_frame = frames[search_index]
            search_path = path + '/' + search_frame + '.jpg'
            search_box = data[search_frame]

            if read_pair:
                if match_range == 'all':
                    template_index = random.randint(0, num_frame - 1)
                elif match_range == 'init':
                    template_index = 0
                elif match_range == 'mix':
                    if random_sys() > 0.5:
                        template_index = 0
                    else:
                        template_index = random.randint(0, num_frame - 1)
                else:
                    template_index = random.randint(max(search_index - match_range, 0),
                                                    min(search_index + match_range, num_frame) - 1)

                template_path = path + '/' + frames[template_index] + '.jpg'
                template_box = data[frames[template_index]]

        search_img = cv2.imread(search_path)
        search_box = np.array(search_box, dtype=np.float32)

        if read_pair:
            if template_path == search_path:
                template_img = search_img
                template_box = search_box
            else:
                template_img = cv2.imread(template_path)
                template_box = np.array(template_box, dtype=np.float32)
            return all_boxes, search_img, search_box, template_img, template_box

        else:
            return all_boxes, search_img, search_box
