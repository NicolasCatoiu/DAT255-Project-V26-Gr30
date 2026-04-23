import keras
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MODEL_DIR, RESULTS_DIR
from models import shallow_model, deep_model
import numpy as np
import tensorflow as tf
import random

tf.keras.backend.set_floatx('float32')
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def loadData():
   
   return {
       "X_train_mel": np.load(f"{DATA_DIR}/X_train_mel.npy"),
        "X_val_mel": np.load(f"{DATA_DIR}/X_val_mel.npy"),
        "X_test_mel": np.load(f"{DATA_DIR}/X_test_mel.npy"),
        "X_train_mfcc": np.load(f"{DATA_DIR}/X_train_mfcc.npy"),
        "X_val_mfcc": np.load(f"{DATA_DIR}/X_val_mfcc.npy"),
        "X_test_mfcc": np.load(f"{DATA_DIR}/X_test_mfcc.npy"),
        "y_train": np.load(f"{DATA_DIR}/y_train.npy"),
        "y_val": np.load(f"{DATA_DIR}/y_val.npy"),
        "y_test": np.load(f"{DATA_DIR}/y_test.npy")
    }
   
def train_model(model, X_train, y_train, X_val, y_val, lr=0.001, batch_size=32, epochs=50, experiment_name="experiment"):
   
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])
    
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"{experiment_name}.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(    
        monitor='val_loss',
        factor=0.5,                        
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
        ]
    
    print(f"\n{'='*50}")
    print(f"Experiment: {experiment_name}")
    print(f"Learning rate: {lr} | Batch size: {batch_size}")
    print(f"Input shape: {X_train.shape[1:]}")
    print(f"{'='*50}\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1)
    
    np.save(os.path.join(RESULTS_DIR, f"{experiment_name}_history.npy"), history.history)
    print(f"Best validity accuracy: {max(history.history['val_accuracy']):.4f}")
    
    return model, history
    

if __name__ == "__main__":
    data = loadData()
    
    baseline_experiments = [
        {
            "name": "shallow_model-melSpectrogram",
            "model": shallow_model(input_shape=(64, 101, 1), num_classes=10),
            "X_train": data["X_train_mel"],
            "X_val": data["X_val_mel"],
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "name": "shallow_model-mfcc",
            "model": shallow_model(input_shape=(40, 101, 1), num_classes=10),
            "X_train": data["X_train_mfcc"],
            "X_val": data["X_val_mfcc"],
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        { #Denne modellen er best
            "name": "deep_model-melSpectrogram",
            "model": deep_model(input_shape=(64, 101, 1), num_classes=10),
            "X_train": data["X_train_mel"],
            "X_val": data["X_val_mel"],
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "name": "deep_model-mfcc",
            "model": deep_model(input_shape=(40, 101, 1), num_classes=10),
            "X_train": data["X_train_mfcc"],
            "X_val": data["X_val_mfcc"],
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        {
            "name": "deep_model-mel-lr0.0001",
            "model": deep_model(input_shape=(64, 101, 1), num_classes=10),
            "X_train": data["X_train_mel"],
            "X_val": data["X_val_mel"],
            "learning_rate": 0.01,
            "batch_size": 32,
        }
    ]
    
    experiments = [
        {
            "name": "best_baselineModel_augmented",
            "model": deep_model(input_shape=(64, 101, 1), num_classes=11),
            "X_train": np.load(f"{DATA_DIR}/X_train_mel_aug11.npy"),
            "X_val":   np.load(f"{DATA_DIR}/X_val_mel_11class.npy"),
            "y_train": np.load(f"{DATA_DIR}/y_train_aug11.npy"),
            "y_val":   np.load(f"{DATA_DIR}/y_val_11class.npy"),
            "learning_rate": 0.001,
            "batch_size": 32,
        }
    ]
    
    results={}
    for exp in experiments:
        model, history = train_model(
            model=exp["model"],
            X_train=exp["X_train"],
            y_train=exp.get("y_train", data["y_train"]),
            X_val=exp["X_val"],
            y_val=exp.get("y_val", data["y_val"]),
            lr=exp["learning_rate"],
            batch_size=exp["batch_size"],
            experiment_name=exp["name"]
        )
        results[exp["name"]] = max(history.history['val_accuracy'])
    
    print("\n" + "="*50)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*50)
    for name, acc in results.items():
        print(f"{name:<30} val_accuracy: {acc:.4f}")
        
        
        
    
    