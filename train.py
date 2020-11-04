import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from dotenv import load_dotenv, find_dotenv
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TerminateOnNaN
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from .models import build_simple_conv_net

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR.joinpath('logs', f'{datetime.now().strftime("%m-%d-%Y_%H:%M:%S")}')

WEIGHTS_OUTPUT_PATH = LOGS_DIR.joinpath('ep{epoch:03d}-loss{loss:.4f}-val_loss{val_loss:.4f}.h5')
MODEL_OUTPUT_PATH = LOGS_DIR.joinpath('trained_model.h5')
DEFAULT_MODEL_INPUT_SIZE = '300x300'

ENV_FILE = find_dotenv()
if ENV_FILE:
    load_dotenv(ENV_FILE)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_input_size', required=False, default=DEFAULT_MODEL_INPUT_SIZE,
                        help="Model image input size")
    parser.add_argument('--val_split', type=float, required=False, default=0.1, help="Train/val split")
    parser.add_argument('--classes_path', required=True, help='Path to class file')
    parser.add_argument('--batch_size', type=int, required=False, default=128,
                        help="Data batch size")
    parser.add_argument('--learning_rate', type=float, required=False, default=0.001,
                        help="Learning rate")
    parser.add_argument('--init_epoch', type=int, required=False, default=0,
                        help="Initial training epochs for fine-tune")
    parser.add_argument('--total_epochs', type=int, required=False, default=30,
                        help="Total training epochs")

    return parser.parse_args()


def main(args):
    w, h = args.model_input_size.split('x')
    input_shape = np.asarray([h, w, 3], dtype=int)
    checkpoint = ModelCheckpoint(str(WEIGHTS_OUTPUT_PATH),
                                 monitor='val_loss',
                                 verbose=1,
                                 save_weights_only=True,
                                 save_best_only=True,
                                 period=1)
    logging = TensorBoard(log_dir=str(LOGS_DIR))
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1, cooldown=0, min_lr=1e-7)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=300, verbose=1)
    terminate_on_nan = TerminateOnNaN()
    callbacks = [logging, checkpoint, reduce_lr, early_stopping, terminate_on_nan]

    optimizer = Adam(lr=args.learning_rate)
    model = build_simple_conv_net(input_shape)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    initial_epoch = args.init_epoch
    epochs = args.total_epochs - initial_epoch
    assert epochs >= 1

    train_datagen = ImageDataGenerator(rescale=1 / 255)
    test_datagen = ImageDataGenerator(rescale=1 / 255)

    train_generator = train_datagen.flow_from_directory(
        './data/training/',
        target_size=input_shape[:2],
        batch_size=args.batch_size,
        class_mode='binary'
    )

    validation_generator = test_datagen.flow_from_directory(
        './data/validation/',
        target_size=input_shape[:2],
        batch_size=args.batch_size,
        class_mode='binary'
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=callbacks
    )

    model.save(MODEL_OUTPUT_PATH)


if __name__ == '__main__':
    K.clear_session()
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
    try:
        args = parse_args()
        main(args)
    except KeyboardInterrupt:
        K.clear_session()
        print('Interrupted by user...')
        sys.exit(0)
