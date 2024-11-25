import tensorflow as tf, keras, numpy as np
import keras.src.backend as k
from keras import layers, Model
from .base_model import BaseModel
from .model_factory import register_model
from utils import Config

@register_model('dueling-mpl', 'dueling mpl')
class DuelingMPLModel(BaseModel):

    @classmethod
    def create_model(cls, input_shape, num_actions, **config) -> Model:
        class AdvantageNormalizationLayer(layers.Layer):
            def call(self, inputs):
                # Normalizar la ventaja restando la media
                return inputs - tf.reduce_mean(inputs, axis=1, keepdims=True)
            
        default_dense_layers = [64, 64]
        dense_layers = Config.get_layers_config(config, default_dense_layers)

        inputs = keras.Input(shape=input_shape)
        x = inputs
        for layer_config in dense_layers:
            x = layers.Dense(**layer_config)(x)
        
        # Ramas para valor y ventaja
        state_value = layers.Dense(1, activation=None)(x)
        action_advantage = layers.Dense(num_actions, activation=None, kernel_initializer="glorot_uniform")(x)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        advantage_normalized = AdvantageNormalizationLayer()(action_advantage)
        
        q_values = layers.Add()([state_value, advantage_normalized])

        model = keras.Model(inputs=inputs, outputs=q_values)
        
        return model