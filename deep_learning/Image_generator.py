# -*- coding: utf-8 -*-

'''
My system is windows7 - anaconda3 , ubuntu16.04 anaconda2.
To run this script , you need to use keras 2.0 tensorflow backend.
You need to modify your system to run this script, Considering suitable environment of your system.
Refer to below ' SETTING ' for detail.
If this script doesn't run properly, check your system one more time.
ex) pip list
    pip install tensorflow
    pip install keras
http://gusrb.tistory.com
@ Example image-data directory structure
photos(root dir)  <- INPUTDIR
    - dog (sub dir)
    - cat
    - pig
    - ...
'''

import multiprocessing, os, time
from keras.preprocessing.image import ImageDataGenerator

''' SETTING '''
INPUT_DIR = r'C:\Users\Jonkim\PycharmProjects\pythonproj\U_turn_images/' # your image data directory/
#여기 디렉토리가 사진 폴더의 상위 폴더여야함 (지금 INPUT_DIR 폴더 안에 u_turn폴더가 있고 그 안에 사진있음
OUTPUT_DIR = r'C:\Users\Jonkim\PycharmProjects\pythonproj\U_turn_output/'  # output images data directory/
IMAGE_FORMAT = 'png'  # output image format (jpeg, png)
FILE_NAME = 'pre'  # output image file name pre***.jpeg
IMAGE_SIZE = (299, 299)  # output image size
P_NUM = multiprocessing.cpu_count() # number of core
END_POINT = 10  # Number of images to be generated per image
# batch_size = Number of images in 1 label
# The batch size is calculated automatically.

# Set image modify
# See keras api documentation
# https://keras.io/preprocessing/image/
train_data = ImageDataGenerator(
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    channel_shift_range=0.1,
    fill_mode='nearest',
    horizontal_flip=False,
    vertical_flip=False,
    rescale=None,
    preprocessing_function=None)

''' Execution '''
def check_start():
    global labels
    labels= os.listdir(INPUT_DIR)
    global labels_cnt
    labels_cnt = len(labels)

    no_image = []
    try:
        if not (os.path.isdir(INPUT_DIR)):
            print("Error : Not a directory.")
            print(INPUT_DIR)
            return False
        else:
            if not (labels):
                print('\nError : Input directory is empty.')
                print(INPUT_DIR)
                return False
            else:
                for name in labels:
                    if (len(os.walk(INPUT_DIR + name).__next__()[2]) == 0):
                        no_image.append(name)
        if (no_image):
            print('\nError : There are no images in the sub directory.')
            for name in no_image:
                print('- ' + name)
                return False
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
    except :
        print('\nError : It is not the root images directory.')
        print("Check the 'INPUT_DIR' and images directory structure.")
        print(INPUT_DIR)
        return False
        exit()
    else :
        return True

def ready_dir():
    # looking for labels
    try :
        print('\nLooking for labels ...')
        for folder_name in labels:
            print(folder_name)
            os.mkdir(OUTPUT_DIR + folder_name)
        print('\n%d Labels' % labels_cnt)
        print('... Completed ...\n')
    except :
        print('Error : Failed to find and create label.')
        exit()

def gen(folder_name):
    i = 0
    batch_cnt = len(os.walk(INPUT_DIR + folder_name).__next__()[2])
    print('- ' + folder_name + ' start ...')
    try:
        for name in train_data.flow_from_directory(
                directory=INPUT_DIR,
                target_size=IMAGE_SIZE,
                batch_size=batch_cnt,
                save_to_dir=OUTPUT_DIR + folder_name,
                save_format=IMAGE_FORMAT,
                save_prefix=folder_name,
                classes=[folder_name]):
            i += 1
            if i > END_POINT:
                print('-- '+folder_name + ' end ...')
                break
    except Exception as e:
        print('\nError : Image generate error !')
        print(e)
        exit()

def gen_run():
    if P_NUM > labels_cnt:
        core = labels_cnt
    else:
        core = P_NUM

    try:
        p = multiprocessing.Pool(core)
        p.map_async(gen, labels).get()
    except Exception as e:
        print('\nError : Process execution error !')
        print(e)
        exit()

def check_end():
    new_labels = os.listdir(OUTPUT_DIR)
    if (labels == new_labels):
        print('\nAll images generated !')
    else:
        error_dir = list(set(labels) - set(new_labels))
        print('\nError : Some images were not generated. Please check the Input/Output directory.')
        for dir in error_dir:
            print(dir)

if __name__ == '__main__':
    start_time = time.time()

    if(check_start()) :
        ready_dir()
        gen_run()
        check_end()
        running_time = time.time() - start_time
        print('RUNNING TIME : %.2f sec' % running_time)
    else:
        exit()