import tensorflow as tf
from tensorflow.keras import layers, models

def create_and_save_dummy_model():

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    model.save('skin_cancer_model.h5')

    print("Model created successfully!")

if __name__ == '__main__':
    create_and_save_dummy_model()