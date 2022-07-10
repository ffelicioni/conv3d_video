import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

def df_train_reordenar(df_train, cant_train,batch):
    df_resto=df_train.copy()
    df_train_batch=pd.DataFrame([])
    for i in range(batch-1):
        X_train_index, X_test_index, y_train, y_test = train_test_split(df_resto.index, df_resto.ID, test_size=cant_train//batch,random_state=0)
        df_seleccion=df_resto.loc[sorted(X_test_index)]   #dataset seleccionado de x videos
        df_resto=df_resto.loc[sorted(X_train_index)]      #sigue iterando en el resto  
        df_train_batch=pd.concat([df_train_batch,df_seleccion],axis=0)    #ordeno el dataset_batch
  

    df_train_batch=pd.concat([df_train_batch,df_resto],axis=0)    #agrego la ultima parte
    return df_train_batch