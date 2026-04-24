import tensorflow as tf
from tensorflow.keras import layers, Model


class AttentionMIL(Model):
    def __init__(self, feature_dim=512, hidden_dim=256, num_classes=1):
        super(AttentionMIL, self).__init__()

        # Moduł uwagi (Attention Network)
        self.attention_dense1 = layers.Dense(hidden_dim, activation='tanh')
        self.attention_dense2 = layers.Dense(1)

        # Klasyfikator końcowy
        self.classifier = layers.Dense(num_classes, activation='sigmoid')

    def call(self, inputs):
        A = self.attention_dense1(inputs)
        A = self.attention_dense2(A)
        A = tf.transpose(A)

        # Normalizacja wag (Softmax)
        A = tf.nn.softmax(A, axis=1)

        # Agregacja z uwagą (Attention Pooling)
        M = tf.matmul(A, inputs)

        # Klasyfikacja całego szkiełka
        Y_prob = self.classifier(M)

        return Y_prob, A