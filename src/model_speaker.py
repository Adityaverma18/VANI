# src/model_speaker.py
from keras import layers, Model, Input
from keras.optimizers import Adam

def build_speaker_model(time_steps, features, n_speakers, lr=1e-4):
    inp = Input(shape=(time_steps, features), name="input_feat")

    x = layers.Conv1D(256, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    # Self-attention
    att = layers.Attention()([x, x])  # shape (batch, time, units)
    x = layers.Flatten()(att)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(n_speakers, activation="softmax", name="speaker_out")(x)

    model = Model(inputs=inp, outputs=out, name="speaker_model")
    model.compile(optimizer=Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model
