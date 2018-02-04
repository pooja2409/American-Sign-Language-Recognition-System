import timeit

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm
from scipy import misc
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
tf.reset_default_graph()
TRAIN_DIR ='C:/Users/pooja/Documents/new8/trainABC'
TEST_DIR ='C:/Users/pooja/Documents/new8/testABC'
IMG_SIZE = 200
LR = 1e-3
MODEL_NAME = 'quickest.model'.format(LR, '2conv-basic')
def get_num(x):
    return int(''.join(ele for ele in x if ele.isdigit()))

def label_img(img):
    word = img.split('.')[-2]
    word_label = word[0]
    if word_label == 'A': return [1,0,0,0,0,0,0,0,0,0,0,0]
    elif word_label == 'B': return [0,1,0,0,0,0,0,0,0,0,0,0]
    elif word_label == 'C': return [0,0,1,0,0,0,0,0,0,0,0,0]
    elif word_label == 'D': return [0,0,0,1,0,0,0,0,0,0,0,0]
    elif word_label == 'E': return [0,0,0,0,1,0,0,0,0,0,0,0]
    elif word_label == 'F': return [0,0,0,0,0,1,0,0,0,0,0,0]
    elif word_label == 'G': return [0,0,0,0,0,0,1,0,0,0,0,0]
    elif word_label == 'H': return [0,0,0,0,0,0,0,1,0,0,0,0]
    elif word_label == 'I': return [0,0,0,0,0,0,0,0,1,0,0,0]
    elif word_label == 'J': return [0,0,0,0,0,0,0,0,0,1,0,0]
    elif word_label == 'K': return [0,0,0,0,0,0,0,0,0,0,1,0]
    elif word_label == 'L': return [0,0,0,0,0,0,0,0,0,0,0,1]



def create_train_data():
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img= misc.imread(path)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data

def process_test_data():
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        label = label_img(img)
        path = os.path.join(TEST_DIR,img)
        imgnum = img.split('.')[-2]
        img_num=get_num(imgnum)
        img= misc.imread(path)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num,np.array(label)])
        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data
train_data= create_train_data()
print(len(train_data))
train = train_data[:-200]
test = train_data[-200:]

X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]
test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]
X = X.reshape([-1, 200, 200, 1])
test_x = test_x.reshape([-1, 200, 200, 1])

convnet = input_data(shape=[None, 200, 200, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 12, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')


model = tflearn.DNN(convnet, tensorboard_dir='log')

model = tflearn.DNN(convnet, tensorboard_dir='log')
if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')


model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

test_data = process_test_data()

x = len(test_data)
print(x)
fig=plt.figure()
count=[0]*12
countfp=[0]*12
real=[0]*12
for num,data in enumerate(test_data[:x]):

    
    img_num = data[1]
    img_data = data[0]
    label=data[2]
    #y = fig.add_subplot(4,6,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]

    if np.argmax(model_out) == 0:
        countfp[0]+=1
    elif np.argmax(model_out) == 1:
        countfp[1]+=1
    elif np.argmax(model_out) == 2:
        countfp[2]+=1
    elif np.argmax(model_out) == 3:
        countfp[3]+=1
    elif np.argmax(model_out) == 4:
        countfp[4]+=1
    elif np.argmax(model_out) == 5:
        countfp[5]+=1
    elif np.argmax(model_out) == 6:
        countfp[6]+=1
    elif np.argmax(model_out) == 7:
        countfp[7]+=1
    elif np.argmax(model_out) == 8:
        countfp[8]+=1
    elif np.argmax(model_out) == 9:
        countfp[9]+=1
    elif np.argmax(model_out) == 10:
        countfp[10]+=1
    elif np.argmax(model_out) == 11:
        countfp[11]+=1

    
    if np.argmax(model_out) == 0 and np.argmax(model_out)==np.argmax(label):
        str_label='A'
        count[0]+=1
    elif np.argmax(model_out) == 1 and np.argmax(model_out)==np.argmax(label):
        str_label='B'
        count[1]+=1
    elif np.argmax(model_out) == 2 and np.argmax(model_out)==np.argmax(label):
        str_label='C'
        count[2]+=1
    elif np.argmax(model_out) == 3 and np.argmax(model_out)==np.argmax(label):
        str_label='D'
        count[3]+=1
    elif np.argmax(model_out) == 4 and np.argmax(model_out)==np.argmax(label):
        str_label='E'
        count[4]+=1
    elif np.argmax(model_out) == 5 and np.argmax(model_out)==np.argmax(label):
        str_label='F'
        count[5]+=1
    elif np.argmax(model_out) == 6 and np.argmax(model_out)==np.argmax(label):
        str_label='G'
        count[6]+=1
    elif np.argmax(model_out) == 7 and np.argmax(model_out)==np.argmax(label):
        str_label='H'
        count[7]+=1
    elif np.argmax(model_out) == 8 and np.argmax(model_out)==np.argmax(label):
        str_label='I'
        count[8]+=1
    elif np.argmax(model_out) == 9 and np.argmax(model_out)==np.argmax(label):
        str_label='J'
        count[9]+=1
    elif np.argmax(model_out) == 10 and np.argmax(model_out)==np.argmax(label):
        str_label='K'
        count[10]+=1
    elif np.argmax(model_out) == 11 and np.argmax(model_out)==np.argmax(label):
        str_label='L'
        count[11]+=1


    if np.argmax(label) == 0:
        str_label='A'
        real[0]+=1
    elif np.argmax(label) == 1:
        str_label='B'
        real[1]+=1
    elif np.argmax(label) == 2:
        str_label='C'
        real[2]+=1
    elif np.argmax(label) == 3:
        str_label='D'
        real[3]+=1
    elif np.argmax(label) == 4:
        str_label='E'
        real[4]+=1
    elif np.argmax(label) == 5:
        str_label='F'
        real[5]+=1
    elif np.argmax(label) == 6:
        str_label='G'
        real[6]+=1
    elif np.argmax(label) == 7:
        str_label='H'
        real[7]+=1
    elif np.argmax(label) == 8:
        str_label='I'
        real[8]+=1
    elif np.argmax(label) == 9:
        str_label='J'
        real[9]+=1
    elif np.argmax(label) == 10:
        str_label='K'
        real[10]+=1
    elif np.argmax(label) == 11:
        str_label='L'
        real[11]+=1

    
    #y.imshow(orig,cmap='gray')
    #plt.title(str_label)
    #y.axes.get_xaxis().set_visible(False)
    #y.axes.get_yaxis().set_visible(False)
print(count)
print(real)
print(countfp)
total=0
for i in range(0,12):
    if (i == 17):
        continue
    total += count[i]
    print("Accuracy of "+chr(i+65)+" = "+str(count[i]/real[i]))
#print("Accuracy of the model is "+str(total/x))
#stop = timeit.default_timer()
#print('The time of execution is:')
#print(stop - start) 
plt.show()



