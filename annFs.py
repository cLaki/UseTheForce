import os.path as path

import cv2
import numpy as np
from PIL import Image
from keras.callbacks import ModelCheckpoint
from keras.layers.core import Dense  # , Activation
from keras.models import Sequential
from keras.optimizers import SGD

import regionFs as regF
import simpleFs as simF


def create_ann():
    # Implementirati vestacku neuronsku mrezu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    # Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.

    ann = Sequential()
    # Postaviti slojeve neurona mreze 'ann'
    # ulaz (24x24 dimenzija slike) i medjusloj
    ann.add(Dense(input_dim=1024, output_dim=128, init="glorot_uniform", activation='sigmoid'))
    # medjusloj i izlaz (32 slike)
    ann.add(Dense(input_dim=128, output_dim=32, init="glorot_uniform", activation='sigmoid'))

    return ann


def train_ann(ann, inputs, outputs):
    inputs = np.array(inputs, np.float32)
    outputs = np.array(outputs, np.float32)

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    ann.compile(loss='categorical_crossentropy', optimizer=sgd)

    callback = SaverCallback("ann_training/weights.epoch{epoch:02d}.h5")
    # ModelCheckpoint('weights.epoch{epoch:02d}.hdf5', monitor='val_loss', verbose=0, save_best_only=False, mode='auto')

    # obucavanje neuronske mreze
    ann.fit(inputs, outputs, nb_epoch=500, batch_size=1, verbose=0, shuffle=False, show_accuracy=False,
            callbacks=[callback])
    # ann.fit(inputs, outputs, nb_epoch=500, batch_size=1, verbose=0, shuffle=False, show_accuracy=False)

    return ann


def load_training_image():
    image = cv2.imread('ann_training/image/fist_state.png')

    return image


def prepare_training_data(merge_images=True):
    num_full = 1
    num_clench = 1
    num_thumb = 1
    hand_state = []
    image_files = []

    while num_full < 11:
        image_files.append('ann_training/image/left_full (' + str(num_full) + ').png')
        image_files.append('ann_training/image/right_full (' + str(num_full) + ').png')
        hand_state.append('left_full_' + str(num_full))
        hand_state.append('right_full_' + str(num_full))
        num_full += 1
    while num_clench < 3:
        image_files.append('ann_training/image/left_clench (' + str(num_clench) + ').png')
        image_files.append('ann_training/image/right_clench (' + str(num_clench) + ').png')
        hand_state.append('left_clench_' + str(num_clench))
        hand_state.append('right_clench_' + str(num_clench))
        num_clench += 1
    while num_thumb < 5:
        image_files.append('ann_training/image/left_thumb (' + str(num_thumb) + ').png')
        image_files.append('ann_training/image/right_thumb (' + str(num_thumb) + ').png')
        hand_state.append('left_thumb_' + str(num_thumb))
        hand_state.append('right_thumb_' + str(num_thumb))
        num_thumb += 1

    if merge_images:
        images = map(Image.open, image_files)
        widths, heights = zip(*(i.size for i in images))

        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset, 0))
            x_offset += im.size[0]

        new_im.save('ann_training/image/fist_state.png')

    return hand_state


def think_and_decide(ann, img_bin, img_orig):
    # img_orig, shape, rectangle, distances = regF.select_roi(img.copy(), img_bin)
    # _, shape, rectangle, _ = regF.select_roi(None, img_bin)
    shape, _ = regF.select_roi_new(img_orig, img_bin)

    if not shape:
        return

    inputs = simF.prepare_for_ann(shape)
    results = ann.predict(np.array(inputs, np.float32))

    return results


class SaverCallback(ModelCheckpoint):
    def on_train_begin(self, logs={}):
        i = 499
        while i != 489:
            filepath = self.filepath.format(epoch=i)
            if path.isfile(filepath):
                self.model.load_weights(filepath)
                self.model.stop_training = True
                break
            else:
                i -= 1
