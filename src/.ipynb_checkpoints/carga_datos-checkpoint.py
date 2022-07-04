import tensorflow as tf
from tensorflow import keras
import numpy as np

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