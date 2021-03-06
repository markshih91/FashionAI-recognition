import sys
import pandas as pd
import cv2
from tqdm import tqdm
from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.optimizers import *
from keras.applications import *
from keras.applications.inception_v3 import preprocess_input
# from keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF


DATA = "0408"
WIDTH = 299
TRAIN_DIR = "/home/shuai_shi/Documents/FashionAI-data/train/"
TEST_DIR = "/home/shuai_shi/Documents/FashionAI-data/test/"
RESULT_DIR = "/home/shuai_shi/Documents/FashionAI-data/result/"
MODELS_DIR = "/home/shuai_shi/Documents/FashionAI-data/model/"
BATCH_SIZE = 16
EPOCHS = 80
CLASSES = ['neck_design_labels', 'collar_design_labels', 'skirt_length_labels', 'pant_length_labels',
           'lapel_design_labels', 'coat_length_labels', 'sleeve_length_labels',
           'neckline_design_labels']


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def con_bright(img, a, g):
    h, w, ch = img.shape
    img2 = np.zeros([h, w, ch], img.dtype)
    return cv2.addWeighted(img, a, img2, 1 - a, g)


df_train = pd.read_csv(TRAIN_DIR + 'label.csv', header=None)
df_train.columns = ['image_id', 'class', 'label']

for ii in range(7, 8):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    session = tf.Session(config=config)

    cur_class = CLASSES[ii]
    df_load = df_train[(df_train['class'] == cur_class)].copy()
    df_load.reset_index(inplace=True)
    del df_load['index']
    prefix_cls = cur_class.split('_')[0]

    file = open((RESULT_DIR + '{}_' + DATA + '.log').format(prefix_cls), 'w')
    sys.stdout = file


    print('{0}: {1}'.format(cur_class, len(df_load)))
    print(df_load.head())

    print(df_load[(df_load.index == 2)])

    n = len(df_load)
    n_class = len(df_load['label'][0])
    X = np.zeros((n * 4, WIDTH, WIDTH, 3), dtype=np.uint8)
    y = np.zeros((n * 4, n_class), dtype=np.uint8)

    pos = True
    for i in tqdm(range(n)):
        tmp_label = df_load['label'][i]
        if len(tmp_label) > n_class:
            print(df_load['image_id'][i])
        img = cv2.resize(cv2.imread(TRAIN_DIR + '{0}'.format(df_load['image_id'][i])), (WIDTH, WIDTH))
        X[i] = img
        y[i][tmp_label.find('y')] = 1

        X[n + i] = cv2.flip(img, 1)
        y[n + i][tmp_label.find('y')] = 1

        if i % 2 == 0:
            angle = np.random.randint(5, 20)
            a = np.random.randint(12, 16) / 10
            g = np.random.randint(-100, -29)
        else:
            angle = np.random.randint(5, 20) * -1
            a = np.random.randint(5, 9) / 10
            g = np.random.randint(29, 101)

        X[2 * n + i] = rotate(img, angle)
        y[2 * n + i][tmp_label.find('y')] = 1


        X[3 * n + i] = con_bright(img, a, g)
        y[3 * n + i][tmp_label.find('y')] = 1


    # 设置session
    KTF.set_session(session)

    cnn_model = InceptionResNetV2(include_top=False, input_shape=(WIDTH, WIDTH, 3), weights='imagenet')
    # resnet50 = ResNet50(include_top=False, input_shape=(WIDTH, WIDTH, 3), weights='imagenet')

    inputs = Input((WIDTH, WIDTH, 3))

    x = inputs
    x = Lambda(preprocess_input, name='preprocessing')(x)
    # x = resnet50(x)
    x = cnn_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(n_class, activation='softmax', name='softmax')(x)

    model = Model(inputs, x)

    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.12, random_state=42)

    adam = Adam(lr=0.001)

    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    checkpointer = ModelCheckpoint(filepath=MODELS_DIR + '{0}.best.h5'.format(prefix_cls), verbose=1,
                                   save_best_only=True)

    h = model.fit(X, y, batch_size = BATCH_SIZE, epochs = EPOCHS,
                  callbacks=[EarlyStopping(patience=5), checkpointer], shuffle=True, validation_split=0.1)


    del(X)
    del(y)
    del (cnn_model)
    del(checkpointer)
    del(h)
    del (model)
    KTF.clear_session()