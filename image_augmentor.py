import os
import cv2
import Augmentor
from tqdm import tqdm
import pandas as pd

TRAIN_DIR = "/home/shuai_shi/Documents/FashionAI-data/train/"
BATCH_SIZE = 16
EPOCHS = 80
CLASS_COUNT = 8000
CLASSES = ['neck_design_labels', 'collar_design_labels', 'neckline_design_labels', 'skirt_length_labels',
           'sleeve_length_labels', 'coat_length_labels', 'lapel_design_labels',
           'pant_length_labels']

df_train = pd.read_csv(TRAIN_DIR + 'label.csv', header=None)
df_train.columns = ['image_id', 'class', 'label']

for i in range(1, 8):

    cur_class = CLASSES[i]
    df_load = df_train[(df_train['class'] == cur_class)].copy()
    df_load.reset_index(inplace=True)

    n = len(df_load)

    # labels_count = {}
    # for j in tqdm(range(n)):
    #     tmp_label = (df_load['label'][j]).replace('m', 'n')
    #     img = cv2.imread(TRAIN_DIR + '{0}'.format(df_load['image_id'][j]))
    #     dirs = (df_load['image_id'][j]).split("/")
    #     path = TRAIN_DIR + 'new_Images/' + cur_class + "/" + tmp_label + "/"
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     cv2.imwrite(path + dirs[-1], img)
    #     if tmp_label not in labels_count.keys():
    #         labels_count[tmp_label] = 1
    #     else:
    #         labels_count[tmp_label] += 1
    #
    # print(labels_count)



    tmp_labels = df_load['label']
    tmp_labels_set = []
    for j in range(len(tmp_labels)):
        label = tmp_labels[j].replace('m', 'n')
        if label not in tmp_labels_set:
            tmp_labels_set.append(label)

    m = len(tmp_labels_set)


    for j in tqdm(range(m)):
        path = TRAIN_DIR + 'new_Images/' + cur_class + "/" + tmp_labels_set[j]
        p = Augmentor.Pipeline(path)
        p.flip_left_right(probability = 0.5)
        p.rotate(probability = 0.7, max_left_rotation = 10, max_right_rotation = 10)
        p.sample(CLASS_COUNT)
