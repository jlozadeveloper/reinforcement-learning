├── .venv/                         # Virtual environment creado con venv
├── data/                          # Directorio para guardar modelos entrenados, checkpoints, logs, gráficos, etc... Se deberán crear los subdirectorios correspondientes para mantenerlo organizado
│   └── experiment_1/
│   │   ├── checkpoints/
│   │   ├── logs/
│   │   ├── graphs/
│   │   └── trained_models/
│   ├── experiment_2/
│   │   └── ...
│
├── config/                        # Directorio para almacenar distintas configuraciones yaml (puede contener subdirectorios para una mejor organización)
│   └── __init__.py                # Archivo para hacer el directorio un paquete Python
│
├── environment/                   # Distintos entornos para entrenar y evaluar agentes, todos se deben ajustar a la api de gymnasium
│   └── __init__.py                # Archivo para hacer el directorio un paquete Python
│   └── rubik_env.py               # Ejemplo de entorno
│
├── model/                         # Modelos de RL implementados con la API funcional de keras: MLP, CNN, RNN, LSTM, Transformers, SVM, 
│   └── __init__.py                # Archivo para hacer el directorio un paquete Python
│   └── base_model.py              # Clase abstracta que define las funciones y estructura que deben exponer los modelos
│   └── dqn_model.py               # Ejemplo de implementación de un modelo DQN
|
├── policy/                        # Implementación de distintas políticas: epsilon-greedy, otra para softmax, UCB, Thompson Sampling, Boltzmann, Determinística con Ruido, Entropía, Meta-Learning
│   └── __init__.py                # Archivo para hacer el directorio un paquete Python
│   └── base_policy.py             # Clase abstracta que define las funciones y estructura que deben exponer las políticas
│   └── epsilon_greedy_policy.py   # Clase que implementa la politica de epsilon greedy
│
├── replay_buffer/                 # Distintas implementaciones del replay buffer: replay buffer standard, Prioritized Experience Replay (PER), Episodic Replay Buffer, Hindsight Experience Replay (HER), Reservoir Sampling Replay Buffer, On-Policy Replay Buffer, Combined Replay Buffer
│   └── __init__.py                # Archivo para hacer el directorio un paquete Python
│   └── base_replay_buffer.py      # Clase abstracta que define las funciones y estructura que deben exponer los diferentes replay buffers
│   └── standard_replay_buffer.py  # Clase que implementa el replay buffer standard
│
├── agent/                         # Agentes entrenados o implementaciones de políticas
│   └── __init__.py                # Archivo para hacer el directorio un paquete Python
│   └── base_agent.py              # Clase abstracta que define las funciones y estructura que deben exponer los agentes
│   └── dqn_agent.py               # Agente que utiliza un modelo DQN
│
├── trainer/                      # Scripts relacionados con el proceso de entrenamiento
│   └── __init__.py                # Archivo para hacer el directorio un paquete Python
│   └── base_trainer.py            # Clase base para los trainer, debe definir las funciones básicas que deben implementar el resto de trainers, y debería implementar la funcionalidad básica compartida como la inicialización, guardado de checkpoints o el logging
│
├── evaluator/                    # Scripts relacionados con el proceso de evaluación de modelos
│   └── __init__.py                # Archivo para hacer el directorio un paquete Python
│   └── base_evaluator.py          # Clase base para los evaluator, debe definir las funciones básicas que deben implementar el resto de evaluators, y debería implementar la funcionalidad básica compartida como la inicialización, guardado de checkpoints o el logging
│
├── utils/                         # Utilidades generales
│   └── __init__.py                # Archivo para hacer el directorio un paquete Python
│   └── plot_utils.py              # Funciones para visualización de recompensas, etc.
│
├── tests/                         # Tests unitarios para diferentes componentes como entornos, agentes etc...
│
├── main.py                        # Script principal para iniciar el entrenamiento
├── requirements.txt               # Dependencias del proyecto
├── .gitignore                     # Archivos y carpetas que git debe ignorar
└── README.md                      # Descripción del proyecto y cómo utilizarlo