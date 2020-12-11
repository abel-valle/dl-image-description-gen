# Generador de descripciones de imágenes

#### Abel Valle Chávez
0231889@up.edu.mx

## Introducción
La intención de este trabajo es poder integrar distintas redes neuronales (NN) y procesamiento de lenguaje natural (NLP). El problema que seleccioné para cumplir con dicha intención se originó en el artículo "Show and Tell: A Neural Image Caption Generator" de Samy Bengio [[1]](#referencias). El objetivo es realizar una implementación para entrenar un modelo de deep learning que permita proporcionar una descripción textual en inglés que vaya de acuerdo a una imagen dada. Las NN que se usan en la implementación son Red Neuronal Convolucional (CNN) complementando con la red neuronal recurrente Long Short Term Memory (LSTM).

## El conjunto de datos
El conjunto de datos utilizado es el de Flickr_8K [2] que cuenta con 8091 imágenes de distintas dimensiones y se ha convertido en un estándar en el problema de descripción textual de imágenes. Dentro de los datos se proporciona un archivo de texto (Flickr8k.token) que contiene el nombre del archivo de imagen y su descripción.

Aunque se incluyen los datos necesarios, se hace referencia a los vículos directos de descarga:
- [flickr8k_dataset](https://nodejs.org/)
- [flickr8k_text](https://nodejs.org/)
- [Github de Jason Brownlee con otros datasets](https://github.com/jbrownlee/Datasets/)

## Las redes neuronales CNN y LSTM
Las CNN son redes neuronales profundas especializadas en procesar datos cuya entrada sea una matrix 2D, por lo tanto son útiles para procesar imágenes. Las CNN se utilizan para clasificación, por ejemplo, para identificar si una imagen es una motocicleta, un avión, un auto, etc. En idea, lo que hace es recorrer la imagen de izquierda a derecha, de arriba hacia abajo obteniento características relevantes y las combina para clasificar. Puede manejar imágenes que estén trasladadas, rotadas o escaladas.

Las redes recurrentes LSTM son apropiadas para problemas donde los eventos suceden en secuencia. Bajo el principio mencionado, es posible predecir la palabra siguiente dato un texto previo. A diferencia de una red recurrente tradicional es que la LSTM puede contener información relevante a través del procesamiento de entradas con un filtro de "olvido", el cual descarta la información no relevante.

## El modelo del generador de descripción de imagen
Para implementar el modelo generador de descripciones se unirán las arquitecturas CNN y LSTM, a dicha unión también se le denomina modelo CNN-RNN.

- La CNN se usa para extraer características de la imagen. Aquí se utiliza el modelo entrenado previamente denominado Xception.
- La LSTM usa la información de la CNN para generar la descripción de la imagen.

## El procesamiento del texto

## 3. Extracción del vector de características de las imágenes
Para esta etapa se aplica la técnica de "transfer learning", que consiste en 
utilizar un modelo previamente entrenado en conjuntos de datos grandes de donde se extraen características que posteriormente utilizamos en nuestro procesamiento. El modelo se reutiliza es Xception, que ha sido entrenado en un conjunto de imágenes que tiene 1000 clases. El modelo se importa directamente de keras.applications. Se observó que el modelo Xception trabaja con imágenes de tamaño (299 x 299 x 3) como entrada, por lo tanto, las imágenes se ajustan al tamaño requerido.

## 5. Tokenización de vocabulario
A cada palabra se le asocia un índice numérico único. Para lo anterior, Keras proporciona la función "tokenizer" que se usa para crear los tokens a partir del vocabulario.

## 6. 

## 7. El modelo CNN-RNN
La definición del modelo consiste en las siguientes partes.

- Extracción de características. La característica extraída de la imagen tiene un tamaño de 2048, la cuál se reduce a 256.
- Procesamiento de secuencia. Una capa embebida maneja la entrada en forma de texto, seguida de una capa LSTM.
- Decoficador. Uniendo la salida de las dos capas anteriores, se procesa mediante una capa densa para tener como salida la predicción final. La capa final contiene el número de unidades igual al tamaño del vocabulario.

| ![Representación visual del modelo CNN-RNN](https://drive.google.com/uc?export=view&id=1-8ViJrXWgdPCkJfHYNXZJqbCmy8JHoXZ) |
|:--:| 
| Figura X. Imagen generada mediante la función plot_model() de keras.utils. |

## 8. Entrenamiento del modelo
Para el entrenamiento del modelo, se utilizan 6000 imágenes generando las secuencias de entrada y salida en batch, ajustándolas al modelo mediante el método model.fit_generator(). El modelo entrenado se guarda en una carpeta para su futuro uso sin necesidad de volver a entrenar.

## Resultados

## Referencias
[1] O. Vinyals, A. Toshev, S. Bengio and D. Erhan, "Show and tell: A neural image caption generator", Proceedings of 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3156-3164, 2015.

[2] P. Young, A. Lai, M. Hodosh, and J. Hockenmaier. From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. In ACL, 2014.

### Códigos de referencia
https://www.kaggle.com/shadabhussain/automated-image-captioning-flickr8
https://www.kaggle.com/wikiabhi/image-caption-generator