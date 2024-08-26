# **Sistema de Validación de Voz**
### Tecnologías a Utilizar
 1. Procesamiento de Series de Audio:
    1. Librosa -> [MFCC](#mel-frequency-cepstral-coefficients)
    2. soundfile -> Nos da control total sobre los archivos de audio al momento de importarlos. A diferencia de Librosa que es una manera más rápida y simple de hacerlo.
    3. PyDub
    4. SciPy
 2. Machine Learning y Deep Learning
    1. TensorFlow / Keras -> CNNs y RNNs
    2. PyTorch
    3. Scikit-Learn
 3. Análisis de Señales
    1. NumPy
    2. SciPy
 4. Seguridad
    1. Crypto
    2. Scikit-Learn
 5. Librerías Extra:
    1. IPython.display -> Audio, Display
    2. wave
    3. pyaudio -> Puede recibir señales del micrófono
    4. sounddevice -> Puede recibir señales del micrófono pero da menos herramientas para tratar que pyaudio
    5. torch, torchaudio





#### Mel-Frequency Cepstral Coefficients
Transformación de dominio de tiempo al dominio de las frecuencias.

 1. División en frames de $\Delta t = 20-40$ms.
 2. Se aplica una [ventana](#ventana) (como la [ventana de Hamming](#venana-de-hamming)).
 3. Aplicamos la transformada de Fourier para encontrar el espectro de frecuencia.
 4. Filtrando [escala de Mel](#escala-de-mel). El espctro se pasa por un cojunto de filtros triangulares dispuestos en una escala de Mel. Los filtros están más densamente espciados a bajas frecuencias y ampliamente a bajas.
 5. Logaritmo: se toma el logaritmo de la energía de cada filtro para obtener una representación logarítmica de las amplitudes.
 6. Aplicamos la Transformada Discreta del Coseno (DCT): Se aplica a los logaritmos de las energías para obtener los coeficientes cepstrales. Los primeros coeficientes capturan las características reelevantes del espectro de frecuencias.

Esto es importante dado que da una *representación compacta*, *reduce la dimensionalidad*, es *robusto* y tiene una *amplia gama de Aplicaciones*.



###### Ventana
Función uitlizada para modificar una porción de una señal. Reduce los efectos discontinuantes en los bordes de los segmentos de una señal (Efectos de Borde o Fenomeno de Gibbs).

###### Venana de Hamming
Tiene la siguiente forma

$$
    \omega (n) = 0.54 + 0.46 \cos{\left( \frac{2\pi n}{N - 1} \right)}
$$

con $n$ el índice de la muestra y $N$ número de muestras en el marco. Suaviza los bordes del marco, reduciendo el espectro de las frecuencias y mejorando la precisión.


###### Escala de Mel
Es una escala perceptual de frecuencias que se basa en como los humanos perciben los sonidos. Es logarítmica a partir de los $1000$Hz, lo que hace que las frecuencias bajas son más perceptibles que las altas.

$$
    m = 2595*log_{10} \left( 1 + \frac{f}{700} \right),
$$

con $f$ en Hz.



### Data Sets
1. [Common Voice](https://commonvoice.mozilla.org/es/datasets): Base de datos abierta de Mozilla.
2. [VoxCeleb](https://robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html): Datos de voz de celebridades.
3. [LibriSpeech](https://www.openslr.org/12): Base de audiolibros en varios idiomas.
   
Se iniciará trabajando con un dataset pequeño de *Common Voice* para realizar las primeras pruebas, luego se trabajará con un dataset mucho más grande de la misma página. Luego ya se incluirán datos adicionales de VoxCeleb. Las últimas pruebas se realizarán con una base local creada con diferentes *"ruidos"* de ambiente para corroborar su utilidad.


### Limpieza de Datos
1. Análisis Inicial: Visualización de la señal de audio (espectrograma).
2. Resampleo y Normalización: Resamplear la señal a una frecuencia de muestreo estándar y normalizar la amplitud de la señal.
3. Eliminación de Ruido de Fondo: Aplicar sustracción de ruido u otra técnica de eliminación de ruido (com el Spectral Gate).
4. Filtrado de Frecuencia: Aplicar filtros paso bajo, paso alto o paso banda para eliminar componentes de frecuencia que no correspondan al mensaje. Y mantener frecuencias reelevantes para la voz humana ($\approx$ $300$Hz y $3400$Hz).
5. Detección de Voz y Segmentación: Utilizar técnicas de detección de actividad de voz (VAD) para segmentar las partes de la señal que contienen habla.
6. Extracción de Características: Extraer las características reelevantes para el reconociemiento de voz, como MFCCs [Mel-Frequency Cepstral Coefficients](#mel-frequency-cepstral-coefficients).

Ya con la data limpia, se puede pasar al modelo de reconocimiento de voz, esto puede realizarse con `tensorflow` `tensorflow_hub`. El entrenamiento del modelo se tratará posteriormente.


### Desarrollo
#### Speaker Identification











#### Material de Apoyo
1. [Python Speech Regocnition Tutorial - Full Course for Beginners](https://www.youtube.com/watch?v=mYUyaKmvu6Y): Las primeras 3 secciones son las más utiles.
2. [Build your own real-time voice command recognition model with TensorFlow](https://www.youtube.com/watch?v=m-JzldXm9bQ): Puede ser útil.
3. [SpeakerIdentification/deeplearning](https://www.youtube.com/watch?v=glAdGQSlm9U)
4. [Speech Recognition w deeplearning](https://www.youtube.com/watch?v=qKz_lmgad3o)