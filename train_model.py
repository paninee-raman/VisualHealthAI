import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_transfer_learning_model(input_shape=(224, 224, 3)):

    print("\n[1/4] Loading MobileNetV2 base model...")

    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze pretrained layers
    base_model.trainable = False

    # Build final model
    model = models.Sequential([

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation='relu'),

        layers.Dropout(0.3),

        layers.Dense(1, activation='sigmoid')

    ])

    # Compile model
    model.compile(

        optimizer=optimizers.Adam(learning_rate=0.001),

        loss='binary_crossentropy',

        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc')
        ]

    )

    model.summary()

    return model


def prep_data_and_train(dataset_dir='dataset', epochs=5, batch_size=32):

    """
    Required folder structure:

    dataset/
        train/
            benign/
            malignant/

        val/
            benign/
            malignant/
    """

    # Check dataset exists
    if not os.path.exists(dataset_dir):

        print(f"\n⚠ Dataset folder '{dataset_dir}' not found.")

        print("Creating folder structure...")

        for split in ['train', 'val']:
            for label in ['benign', 'malignant']:

                os.makedirs(
                    os.path.join(dataset_dir, split, label),
                    exist_ok=True
                )

        print("\nFolders created.")
        print("Add images into the folders and run again.")

        return

    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    print("\n[2/4] Preparing datasets...")

    # Training data augmentation
    train_datagen = ImageDataGenerator(

        rescale=1./255,

        rotation_range=30,

        width_shift_range=0.2,

        height_shift_range=0.2,

        zoom_range=0.2,

        shear_range=0.2,

        horizontal_flip=True,

        vertical_flip=True,

        fill_mode='nearest'

    )

    # Validation data
    val_datagen = ImageDataGenerator(
        rescale=1./255
    )

    # Load training images
    train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='binary'

    )

    # Load validation images
    val_generator = val_datagen.flow_from_directory(

        val_dir,

        target_size=(224, 224),

        batch_size=batch_size,

        class_mode='binary'

    )

    print("\nClass labels:")
    print(train_generator.class_indices)

    # Build model
    model = build_transfer_learning_model()

    print(f"\n[3/4] Training started for {epochs} epochs...\n")

    # Train model
    history = model.fit(

        train_generator,

        validation_data=val_generator,

        epochs=epochs

    )

    print("\n[4/4] Saving trained model...")

    model.save('skin_cancer_model.h5')

    print("\n✅ Model saved successfully as 'skin_cancer_model.h5'")


if __name__ == '__main__':

    prep_data_and_train(
        epochs=20,
        batch_size=32
    )