from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Embedding, Lambda, Concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.backend import abs as ab
from tensorflow.keras.models import Model
from tensorflow.keras import Input


def siamese_model(glove_weights, max_len, lstm_unit, dropout):
    # Get shapes of embedding for embedding layer
    (num_vocab, emb_dim) = glove_weights.shape

    # Create a Bidirectional LSTM layer for understanding meaning of the words.
    # Bidirectional is better for understanding than regular LSTM.
    lstm = Bidirectional(LSTM(lstm_unit, dropout=dropout, recurrent_dropout=dropout))
    # Create embedding layer and use glove embeddings and do not train embeddings
    embed = Embedding(input_dim=num_vocab, output_dim=emb_dim, input_length=max_len, weights=[glove_weights],
                      trainable=False)
    # Create an input layer for sentence one
    input1 = Input(shape=(max_len,))
    # Feed sentence one through embedding
    e1 = embed(input1)
    # Feed the embedded sentence to BiLSTM
    t1 = lstm(e1)

    # Create second input layer for sentence two
    input2 = Input(shape=(max_len,))
    # Feed sentence two through embedding
    e2 = embed(input2)
    # Feed the embedded sentence to BiLSTM
    t2 = lstm(e2)
    # These sentences share same weights for training

    # Subtract the output of the both sentences
    subtract = lambda x: ab(x[0] - x[1])
    sub_layer = Lambda(function=subtract, output_shape=lstm_unit)([t1, t2])
    # Feed it through a Linear layer with sigmoid activation function
    predicts = Dense(1, activation='sigmoid')(sub_layer)

    # Create the model with inputs and output
    model = Model(inputs=[input1, input2], outputs=predicts)

    # Define optimizer, loss function and metrics
    model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model
