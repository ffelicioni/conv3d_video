import os as os
from pathlib import Path

import numpy as np
import math as math

from google.colab.patches import cv2_imshow

import cv2

from skimage import io

import random as random

import shutil


def video_capturing_function(dataset,folder_name):
    for i in sorted(dataset.file_path.index):
        video_name=Path(dataset.file_path[i]).stem
        video_read_path=dataset.file_path[i]
        cap=cv2.VideoCapture(video_read_path)
        stretches_path='/content'
        
        try:
            if not os.path.exists(os.path.join(stretches_path,folder_name)):
                os.mkdir(os.path.join(stretches_path,folder_name))
                
            os.mkdir(os.path.join(stretches_path,folder_name,video_name))
        except:
            print("Folder Already Created")
        
        
        train_write_file=os.path.join(os.path.join(stretches_path,folder_name),video_name)
        cap.set(cv2.CAP_PROP_FPS, 1)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #print(frame_count)
        #frameRate=cap.get(5)
        x=1
        count=0
        while(cap.isOpened()):
            frameId = cap.get(1) #current frame number
            ret, frame = cap.read()

            if (ret != True):
                break
            filename ="frame%d.jpg" % count;count+=1
            frame_grey=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #para escala de gris 
            cv2.imwrite(os.path.join(train_write_file,filename), frame_grey)
        cap.release()
    return print("All frames written in the: "+folder_name+" Folder")


def data_load_function_frames(dataset,directory, frame_ref):
    frames=[]
    for i in sorted(dataset.file_path.index):
        vid_name=Path(dataset.file_path[i]).stem
        vid_dir_path=os.path.join(directory,vid_name)
        frame_count = len(os.listdir(vid_dir_path))  #cantidad de frames 
        
        frames_to_select=[]
        for l in np.arange(0,frame_count):
            frames_to_select.append('frame%d.jpg' % l)

        frames_to_select=resampling_rand(frames_to_select,frame_ref, frame_count)
        #print(frames_to_select)
        vid_data=[]
        for frame in frames_to_select:            
            image=(io.imread(os.path.join(vid_dir_path,frame))/255.0).astype('float32')
            #image=image.resize((250, 250), Image.ANTIALIAS) 
            #datu=np.asarray(image)
            #normu_dat=datu/255
            #vid_data.append(normu_dat)
            vid_data.append(image)
        vid_data=np.array(vid_data)
        frames.append(vid_data)
    return np.array(frames)

def resampling_rand(frames_to_select,frame_ref, frame_count):
    if (frame_count < frame_ref):
        frames_to_select=frames_to_select+[frames_to_select[-1]]*(frame_ref-frame_count)
    else:
        random.seed=0
        new_dic_lis = dict(zip(range(0,len(frames_to_select)), frames_to_select)) 
        frames_to_select=[new_dic_lis[x] for x in sorted(random.sample(new_dic_lis.keys(),frame_ref))]
    return (frames_to_select)

def data_load_frames_save_tf(dataset,directory, frame_ref):
    frames=[]
    for i in sorted(dataset.file_path.index):
        vid_name=Path(dataset.file_path[i]).stem
        vid_dir_path=os.path.join(directory,vid_name)
        frame_count = len(os.listdir(vid_dir_path))  #cantidad de frames 
        
        frames_to_select=[]
        for l in np.arange(0,frame_count):
            frames_to_select.append('frame%d.jpg' % l)

        frames_to_select=resampling_rand(frames_to_select,frame_ref, frame_count)
        #print(frames_to_select)
        vid_data=[]
        for frame in frames_to_select:            
            image=(io.imread(os.path.join(vid_dir_path,frame))/255.0).astype('float32')
            vid_data.append(image)
        vid_data=np.array(vid_data)
        #print(vid_dir_path+'.npy')
        np.save(vid_dir_path+'.npy',vid_data)
        shutil.rmtree(vid_dir_path)
        #frames.append(vid_data)
    return None #frames

def load_data_from_tf(dataset,directory):
    list_tf=[]
    for i in sorted(dataset.file_path.index):
        vid_name=Path(dataset.file_path[i]).stem
        vid_dir_path=os.path.join(directory,vid_name)
        vid_data=np.load(vid_dir_path+'.npy')
        list_tf.append(vid_data)

    #print(np.max(np.array(list_tf)))
    return np.array(list_tf)



class CustomSequence(keras.utils.Sequence):

    def __init__(self, batch_size):
        super().__init__()

        self.batch_size = batch_size
        self.n=3840
        

    def __len__(self):
        # return the number of BATCHES (not samples)
        # In this dataset
        return self.n // self.batch_size

    def __getitem__(self, batch_index):
        # Return BATCH (not a sample) with index `batch_index`

        #Calculate first and last index of samples of the batch
        idx_start = batch_index*self.batch_size
        idx_end = idx_start+self.batch_size
        #select samples
        #batch_x = self.x[start:end,:,:,:]
        #batch_y = self.y[start:end,0]

        X=load_data_from_tf(df_train_batch[idx_start:idx_end],"train_data")
        a=np.array(df_train_batch[idx_start:idx_end].ID)
        y = tf.keras.utils.to_categorical(a-1, num_classes =64)
        #print('salida:',np.max(df_train_batch[idx_start:idx_end].ID)) 

        nf,w,h=X[0].shape
        #print(nf,w,h)
        #print(len(X))
        X.reshape(self.batch_size,nf,w,h,1)
        
        return X, y
    
    
def generator(batch_size):
    n=3840
    while True:
        for i in range(n//batch_size):
            index=i*batch_size
            X=load_data_from_tf(df_train_batch[index:index+batch_size],"train_data")
            #y=df_train_batch[index:index+batch_size].ID
            a=np.array(df_train_batch[index:index+batch_size].ID)
            y = tf.keras.utils.to_categorical(a-1, num_classes =64)
      

            nf,w,h=X[0].shape
            #print(nf,w,h)
            #print(len(X))
            X.reshape(batch_size,nf,w,h,1)
            #return X, y

            yield X,y