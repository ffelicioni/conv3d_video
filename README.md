# Trabajo Práctico Taller de tesis 1 - Deep Learning

## Aplicación de redes neuronales para clasificación de lenguaje de señas a partir de vídeos 

La librería LSA64 [[1]](#1) tiene un conjunto de 4200 vídeos en el cual un grupo de 10 personas fueron realizando un total de 64 distintas señas diferentes. Un conjunto de estas filmaciones se realizaron al aire libre con luz natural, otras en el interior con iluminación artificial de forma de tener diferente iluminación en las imágenes.
Las señas elegidas fueron seleccionadas entre las más comunes del lenguaje de señas en Argentina.
Librería de videos [LSA64: A Dataset for Argentinian Sign Language](http://facundoq.github.io/datasets/lsa64/)

Se utilizan modelos convolucionales (conv3D) para Deep Learning de la librería [Tensorflow/Keras](https://keras.io/api/layers/convolution_layers/convolution3d/) [[2]](#2) . Este trabajo tiene como antecedentes [[3]](#3). Otras referencias de interés [[5]](#5) y [[6]](#6)
 
Este trabajo de Taller de Tesis 1 se relaciona principalmente con la materia Redes Neuronales.

El trabajo realizado está disponible en [link a github](https://github.com/ffelicioni/conv3d_video/)
La notebook de google colab [[4]](#4) está accesible en [link a notebook](https://github.com/ffelicioni/conv3d_video/blob/main/clasificar_hands.ipynb)

## References

<a id="1">[1] </a>
Ronchetti, Franco and Quiroga, Facundo and Estrebou, Cesar and Lanzarini, Laura and Rosete, Alejandro, (2016).
LSA64: A Dataset of Argentinian Sign Language.
XX II Congreso Argentino de Ciencias de la Computación (CACIC).

<a id="2">[2]</a>
Chollet, F. (2021).
Deep Learning with Python, Second Edition
[link](https://books.google.com.ar/books?id=XHpKEAAAQBAJ)

<a id="3">[3]</a>
Iván Mindlin, Franco Ronchetti (2021).
Reconocimiento de Lengua de Señas con Redes Neuronales Recurrentes.
Tesina de Licenciatura. Facultad de Informática. Universidad Nacional de La Plata

<a id="4">[4]</a>
Felicioni, F. (2022).
Notebook colab. Materia Taller de Tesis 1 de la Maestría en Explotación de Datos y Descubrimiento del Conocimiento.

<a id="5">[5]</a>
Koller, O. (2020). 
Quantitative survey of the state of the art in sign language recognition. 
arXiv preprint arXiv:2008.09918.

<a id="6">[6]</a>
Bragg, D., Koller, O., Bellard, M., Berke, L., Boudreault, P., Braffort, A., ... & Ringel Morris, M. (2019). 
Sign language recognition, generation, and translation: An interdisciplinary perspective. 
The 21st international ACM SIGACCESS conference on computers and accessibility (pp. 16-31).


