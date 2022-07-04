import matplotlib.gridspec as gridspec

def plot_conv_weight(layer_name,filters,max_filters=128,cols=12):
    h,w,c=filters.shape
    if c>max_filters:
        filters=filters[:,:,max_filters]
        c = max_filters
    # calcular cant de filas
    rows=( c // cols ) + int((c % cols)>0)
    # grilla de imágenes
    f=plt.figure(figsize=(cols,rows),dpi=100)
    gs = gridspec.GridSpec(rows, cols,wspace=0.0001,hspace=0.001)
    # maximo y minimo para dibujar todos en la misma escala
    mi,ma=filters.min(),filters.max()
    for i in range(c):
        ax = plt.subplot(gs[i])
        ax.imshow(filters[:,:,i],cmap="gray",vmin=mi,vmax=ma)
        ax.set_xticks([])
        ax.set_yticks([])
     # poner en blanco los ax que sobran
    for i in range(c,cols*rows):
        ax = plt.subplot(gs[i])
        ax.axis("off")

      

def layer_by_name(model,layer_name):
    for layer in model.layers:
        if layer.name==layer_name:        
            return layer
    raise ValueError(f"Invalid layer name {layer_name}")