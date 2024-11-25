El proyecto deberá contener agentes y trainers

| Algoritmo | Arquitectura de modelo | Características clave |
|----------|----------|----------|
| PPO | MLP, CNN | Estable, eficiente, política gradiente, actualización de política con restricción |
| DDPG | MLP (actor y crítico) | Acciones continuas, actor-crítico, redes gemelas para estabilidad |
| Dueling | Modificación de la arquitectura (valor y ventaja) | Mejora la estimación de valores, separación de valor del estado y ventaja de la acción |
| Actor-Critic | MLP (actor y crítico) | Combina valor y política, flexibilidad en la arquitectura |
| A2C (Advantage Actor-Critic) | MLP (actor y crítico) | Variante de actor-critic que utiliza la ventaja en lugar del valor |
| A3C (Asynchronous Advantage Actor-Critic) | MLP (actor y crítico) | Entrenamiento paralelo en múltiples entornos para acelerar el aprendizaje |
| SARSA | Tabla o red neuronal | Actualiza los valores Q utilizando la acción tomada, on-policy |
| Q-learning | Tabla o red neuronal | Actualiza los valores Q utilizando la acción óptima, off-policy |
| DQN | MLP, CNN | Combina Q-learning con redes neuronales profundas, experiencia replay, target network |
| TD3 (Twin Delayed Deep Deterministic Policy Gradient) | MLP (actor y crítico) | Mejora de DDPG con dos redes críticas y retardo de la actualización de la política |