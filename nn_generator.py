import keras
from sklearn.metrics import confusion_matrix, classification_report
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential, load_model
import os, itertools
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from claptcha import Claptcha
from create_captchas import ch_list
import seaborn as sns

def plot_confusion_matrix(cm, name, char_list, cmap=plt.cm.RdBu):
    #Create the basic matrix.
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, cmap)

    #Add title and Axis Labels
    plt.title(name + ' - ' 'Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    #Add appropriate Axis Scales
    tick_marks = np.arange(len(char_list))
    plt.xticks(tick_marks, char_list, rotation=45)
    plt.yticks(tick_marks, char_list)

    #Add Labels to Each Cell
    thresh_min = 60
    thresh_max = 120

    #Add a Side Bar Legend Showing Colors
    plt.colorbar()
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if ((cm[i, j] >= thresh_min) & (cm[i, j] <= thresh_max))  else "white")

    plt.tight_layout()
    fig.savefig('confusion_matrices/' + name + '.png', bbox_inches='tight', dpi=1920)
    plt.show()


def nn__(X_val, X_train, y_val, y_train, X_test, y_test, activation, epochs, batch, name, nodes = [100, 50], h_layers = 2, plot = True, dropout = [False,False,False,False]):

        nn_ = Sequential()

        nn_.add(Dense(X_train.shape[1], input_shape = (X_train.shape[1], ), activation = activation))

        if h_layers >= 1:
            nn_.add(Dense(nodes[0], activation = activation))
            if dropout[0]:
                nn_.add(Dropout(.2))
        if h_layers >= 2:
            nn_.add(Dense(nodes[1], activation = activation))
            if dropout[1]:
                nn_.add(Dropout(.2))
        if h_layers >= 3:
            nn_.add(Dense(nodes[2], activation = activation))
            if dropout[2]:
                nn_.add(Dropout(.2))
        if h_layers >= 4:
            nn_.add(Dense(nodes[3], activation = activation))
            if dropout[3]:
                nn_.add(Dropout(.2))

        nn_.add(Dense(y_train.shape[1], activation = 'softmax'))

        nn_.summary()
        nn_.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta = 0.001, patience = 15, verbose=1, mode='auto', baseline=None)
        tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

        nn_.fit(X_train, y_train, batch_size = batch, epochs = epochs, verbose = 1, validation_data = (X_val, y_val), callbacks = [early_stopping, tensorboard])

        if plot == True:
            model_val_dict = nn_.history.history
            loss_values = model_val_dict['loss']
            val_loss_values = model_val_dict['val_loss']
            acc_values = model_val_dict['acc']
            val_acc_values = model_val_dict['val_acc']

            epochs_ = range(1, len(loss_values) + 1)
            plt.plot(epochs_, loss_values, 'g', label='Training loss')
            plt.plot(epochs_, val_loss_values, 'g.', label='Validation loss')
            plt.plot(epochs_, acc_values, 'r', label='Training acc')
            plt.plot(epochs_, val_acc_values, 'r.', label='Validation acc')

            plt.title('Training & validation loss / accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        nn_.save('./models_/nn_/' + name + '_' + activation +'_' + str(epochs) + '_' + str(batch) + '_' + str(h_layers) + '_' + '_'.join([str(i) for i in nodes]) + '_' + str(dropout[0]) + '.h5')

        print(nn_.evaluate(X_test, y_test))

        cm = confusion_matrix(nn_.predict_classes(X_test), np.array(y_test).argmax(axis = 1))
        print(cm)
        plot_confusion_matrix(cm, activation + '_' + name, ch_list)

        print(classification_report(nn_.predict_classes(X_test), np.array(y_test).argmax(axis = 1), target_names=ch_list))
        return nn_, cm

if False:

    test_labels = pd.read_csv('labels/test_labels.csv')
    train_labels = pd.read_csv('labels/train_labels.csv')
    val_labels = pd.read_csv('labels/val_labels.csv')

    X_train = np.concatenate(train_labels['train_path'].apply(lambda x: np.array(Image.open(x))[:,:,0].flatten() / 255)).reshape(43400, 784)
    X_val = np.concatenate(val_labels['val_path'].apply(lambda x: np.array(Image.open(x))[:,:,0].flatten() / 255)).reshape(6200, 784)
    X_test = np.concatenate(test_labels['test_path'].apply(lambda x: np.array(Image.open(x))[:,:,0].flatten() / 255)).reshape(12400, 784)

    y_train = np.array(pd.get_dummies(train_labels['train_label']))
    y_val = np.array(pd.get_dummies(val_labels['val_label']))
    y_test = np.array(pd.get_dummies(test_labels['test_label']))

    np.save('./image_arrays/X_val', X_val)
    np.save('./image_arrays/X_train', X_train)
    np.save('./image_arrays/y_val', y_val)
    np.save('./image_arrays/y_train', y_train)
    np.save('./image_arrays/X_test', X_test)
    np.save('./image_arrays/y_test', y_test)

if False:
    X_val = np.load('./image_arrays/X_val.npy')
    X_train = np.load('./image_arrays/X_train.npy')
    y_val = np.load('./image_arrays/y_val.npy')
    y_train = np.load('./image_arrays/y_train.npy')
    X_test = np.load('./image_arrays/X_test.npy')
    y_test = np.load('./image_arrays/y_test.npy')

if False:

    labels = pd.read_csv('train_2.txt', sep=', ', names=['path','label','type'])

    train_labels = labels[labels.type=='train']
    test_labels = labels[labels.type=='test']
    val_labels = labels[labels.type=='val']

    X_train = np.concatenate(train_labels['path'].apply(lambda x: np.array(Image.open(x))[:,:,0].flatten() / 255)).reshape(18144, 1568)
    X_val = np.concatenate(val_labels['path'].apply(lambda x: np.array(Image.open(x))[:,:,0].flatten() / 255)).reshape(2592, 1568)
    X_test = np.concatenate(test_labels['path'].apply(lambda x: np.array(Image.open(x))[:,:,0].flatten() / 255)).reshape(5184, 1568)

    y_train = np.array(pd.get_dummies(train_labels['label']))
    y_val = np.array(pd.get_dummies(val_labels['label']))
    y_test = np.array(pd.get_dummies(test_labels['label']))

    np.save('./multi_ch_arrays/X_val', X_val)
    np.save('./multi_ch_arrays/X_train', X_train)
    np.save('./multi_ch_arrays/y_val', y_val)
    np.save('./multi_ch_arrays/y_train', y_train)
    np.save('./multi_ch_arrays/X_test', X_test)
    np.save('./multi_ch_arrays/y_test', y_test)

if False:
    X_val = np.load('./multi_ch_arrays/X_val.npy')
    X_train = np.load('./multi_ch_arrays/X_train.npy')
    y_val = np.load('./multi_ch_arrays/y_val.npy')
    y_train = np.load('./multi_ch_arrays/y_train.npy')
    X_test = np.load('./multi_ch_arrays/X_test.npy')
    y_test = np.load('./multi_ch_arrays/y_test.npy')


if True:
        def split_target_image(filename, number_of_steps, del_a):
            #import nn_ into function + nn_.predict()
            model = load_model('models_/nn_/nn__917_v3_relu_75_200_4_750_500_250_125_False.h5')

            preds = []

            img_ = np.array(Image.open(filename))
            length_of_image = img_.shape[1]
            for i in range(0, number_of_steps):
                _img = img_[:, int(((length_of_image - del_a) / number_of_steps)*i) : int(((length_of_image - del_a) / number_of_steps)*i + del_a)]
                img = (_img[:,:,0].flatten()).reshape(784,1).T
                A = Image.fromarray(_img)
                pred = model.predict_classes(img)
                preds.append(ch_list[int(pred)])
                print(pred, ch_list[int(pred)])
                A.show()
            sns.countplot(preds)
            plt.show()
            return preds



        test = split_target_image('captcha_output_data2/multi_char_train_data/mq_19.png', 3, 28)
        #test = split_target_image('captcha_output_data/single_char_test_data/A_upper_235.png', 1, 28)

#test_model = nn__(X_val, X_train, y_val, y_train, X_test, y_test, 'relu', epochs=50, batch=100, name='nn__917_v4', nodes = [500,250,125], h_layers = 3, plot = True, dropout = [False,False,False,False])
