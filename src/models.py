from tensorflow import keras
from tensorflow.keras import layers

def shallow_model(input_shape, num_classes):
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, kernel_size=(3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"),
        layers.MaxPooling2D((2,2)),
        
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
        
    ], name = "shallow_model")
    return model

def deep_model(input_shape, num_classes):
    
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ], name="deep_model")
    return model

print("MODEL A — Mel input:")
model_test = shallow_model((64, 101, 1), 10)
model_test.summary()

print("\nMODEL B — Mel input:")
model_test = deep_model(input_shape=(64, 101, 1), num_classes=10)
model_test.summary()