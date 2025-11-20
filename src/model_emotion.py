from keras import layers, Model, Input
from keras.optimizers import Adam

def build_emotion_model(time_steps, features, classes, lr=1e-4):
    inp = Input(shape=(time_steps, features))

    x = layers.Conv1D(256, 5, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Attention()([x, x])     # <---- IMPORTANT

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(classes, activation="softmax")(x)

    model = Model(inp, out)
    model.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
