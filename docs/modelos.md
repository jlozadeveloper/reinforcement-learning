## Modelos

En el entorno, se generarán diferentes tipos de modelos atendiendo a su arquitectura, siendo siempre lo mas customizables posibles por medio de configuración yaml.  
Los modelos que lo permitan o requieran, contendrán diferentes versiones de la misma arquitectura para soportar entoornos con acciones discretas, y entornos con acciones continuas.

Las arquitecturas a tener en cuenta son:

| Arquitectura de Modelo | Descripción | Algoritmos Compatibles | Características | Usos Principales |
|----------|----------|----------|----------|----------|
| **MLP** (Multi-Layer Perceptron) | Perceptrón multicapa | DQN, DDPG, A2C, PPO | Redes neuronales completamente conectadas, forman la base de muchas otras arquitecturas. | Aproximación de funciones, representación de estados y acciones. |
| **CNN** (Convolutional Neural Network) | Red neuronal convolucional | DQN, PPO | Extrae características locales de datos espaciales (imágenes, videos). | Procesamiento de imágenes, visión por computadora en entornos de RL. |
| **RNN** (Recurrent Neural Network) | Red neuronal recurrente | LSTM, GRU, A2C, PPO | Procesa secuencias de datos, tiene memoria a corto plazo. | Juegos secuenciales, control de robots con dinámica variable, procesamiento de lenguaje natural, series temporales, control de sistemas dinámicos. |
| **GRU** (Gated Recurrent Unit) | Unidad recurrente con puerta | A2C, PPO | Variante de RNN más simple que LSTM, manejo de dependencias a largo plazo. | Similar a LSTM, pero con menos parámetros. |
| **LSTM** (Long Short-Term Memory) | Memoria a largo plazo a corto plazo | A2C, PPO | Maneja dependencias a largo plazo, ideal para tareas que requieren recordar información a largo plazo. | Tareas que requieren recordar información durante largos períodos. Procesamiento de lenguaje natural, series temporales, control de robots. |
| **PPO** (Proximal Policy Optimization) | Optimización de política proximal | DQN, A2C, PPO | Algoritmo de política gradiente, actualizaciones de política seguras. | Amplia gama de tareas de RL, especialmente en entornos complejos. |