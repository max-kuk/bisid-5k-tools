import os
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"


from classification_models.keras import Classifiers
import numpy as np
import tensorflow as tf
from keras import backend as K
from datasets import get_sc_dataset
import logging

from keras import ops, layers, optimizers, callbacks, models, mixed_precision
from keras.src.utils import summary_utils
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def get_model_memory_usage(batch_size, model):
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for l in model.layers:
        layer_type = l.__class__.__name__
        if layer_type == "Model" or layer_type == "Functional":
            internal_model_mem_count += get_model_memory_usage(batch_size, l)
        single_layer_mem = 1
        out_shape = l.output.shape
        if type(out_shape) is list:
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = summary_utils.count_params(model.trainable_weights)
    non_trainable_count = summary_utils.count_params(model.non_trainable_weights)

    number_size = 4.0
    if K.floatx() == "float16":
        number_size = 2.0
    if K.floatx() == "float64":
        number_size = 8.0

    total_memory = number_size * (
        batch_size * shapes_mem_count + trainable_count + non_trainable_count
    )
    gbytes = np.round(total_memory / (1024.0**3), 3) + internal_model_mem_count
    return gbytes


def transform_dataset(x):

    # convert to float32
    rgb_image = ops.cast(x["rgb_image"], np.float32)

    # normalize
    rgb_image = rgb_image / 255.0
    rgb_image.set_shape([None, 192, 192, 3])

    return rgb_image, ops.cast(x["class_id"], np.int8)


def get_data(src_dir, batch_size=64):
    train_ds = (
        get_sc_dataset(
            src_dir,
            "train",
            batch_size=batch_size,
            data_type="rgb",
        )
        .map(transform_dataset)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    val_ds = (
        get_sc_dataset(
            src_dir,
            "val",
            batch_size=batch_size,
            data_type="rgb",
        )
        .map(transform_dataset)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return train_ds, val_ds


def train_model(args, shape_size, run_id):

    train_ds, val_ds = get_data(args.src_dir, batch_size=args.batch_size)

    modelPoint, preprocess_input = Classifiers.get(args.model)
    model = modelPoint(
        input_shape=shape_size,
        include_top=False,
        weights=None,
    )
    # x = model.layers[-1].output
    x = layers.GlobalAveragePooling2D()(model.output)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(
        args.num_classes, name="prediction", activation="softmax", dtype="float32"
    )(x)
    model = models.Model(inputs=model.inputs, outputs=x)

    logging.info("Model: {}".format(args.model))
    # log model summary
    logging.info(model.summary())

    # log model memory usage
    logging.info(
        "Model memory usage: {} GB".format(
            get_model_memory_usage(args.batch_size, model)
        )
    )

    model.compile(
        optimizer=optimizers.AdamW(learning_rate=args.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=[
            "accuracy",
        ],
        jit_compile=True,
    )

    callback_list = [
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.95,
            patience=3,
            min_lr=1e-9,
            min_delta=1e-8,
            verbose=args.verbose,
            mode="min",
        ),
        callbacks.CSVLogger(
            os.path.join(
                args.save_dir,
                "history_{}_lr_{}_run_{}.csv".format(
                    args.model, args.learning_rate, run_id
                ),
            ),
            append=True,
        ),
    ]

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        validation_data=val_ds,
        verbose=2,
        callbacks=callback_list,
    )

    return history, model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_dir", type=str, default="preprocessed_data/tfrecords_dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="model name (resnet18, resnet32, resnet50)",
    )
    parser.add_argument("--learning_rate", type=float, default=0.00001)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="paper_logs_2d")
    parser.add_argument("--mixed_precision", type=bool, default=False)
    parser.add_argument("--n_runs", type=int, default=3)
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    if args.mixed_precision:
        policy = mixed_precision.Policy("mixed_float16")
        mixed_precision.set_global_policy(policy)

    # create save_dir if it does not exist
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    shape_size = (192, 192, 3)  # (24, 24, 3)  # (192, 192, 3)

    # set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    acc_scores = []
    f1_scores = []
    prec_scores = []
    rec_scores = []

    for i in range(args.n_runs):
        tf.keras.backend.clear_session()
        logging.info("Training model: {}, run {}".format(args.model, i))

        # with strategy.scope():
        history, model = train_model(args, shape_size, i)

        best_loss = max(history.history["val_loss"])

        logging.info("Training finished. Loss: {}".format(best_loss))

        logging.info("Starting test phase")

        test_ds = (
            get_sc_dataset(
                args.src_dir,
                "test",
                batch_size=args.batch_size,
                data_type="rgb",
            )
            .map(transform_dataset)
            .prefetch(tf.data.experimental.AUTOTUNE)
        )

        y_pred = model.predict(test_ds)

        y_pred = np.argmax(y_pred, axis=1)

        y_true = []
        for x in test_ds:
            y_true.extend(x[1].numpy())
        y_true = np.array(y_true)

        acc = np.round(accuracy_score(y_true, y_pred) * 100, 2)
        f1 = np.round(f1_score(y_true, y_pred, average="macro") * 100, 2)
        prec = np.round(precision_score(y_true, y_pred, average="macro") * 100, 2)
        rec = np.round(recall_score(y_true, y_pred, average="macro") * 100, 2)

        logging.info("Accuracy: {}".format(acc))
        logging.info("F1: {}".format(f1))
        logging.info("Precision: {}".format(prec))
        logging.info("Recall: {}".format(rec))

        acc_scores.append(acc)
        f1_scores.append(f1)
        prec_scores.append(prec)
        rec_scores.append(rec)

        del model
        del history
        del y_pred
        del y_true
        del test_ds

    logging.info("Test phase finished")

    accuracy_mean = np.round(np.mean(acc_scores), 2)
    accuracy_std = np.round(np.std(acc_scores), 2)

    f1_mean = np.round(np.mean(f1_scores), 2)
    f1_std = np.round(np.std(f1_scores), 2)

    precision_mean = np.round(np.mean(prec_scores), 2)
    precision_std = np.round(np.std(prec_scores), 2)

    recall_mean = np.round(np.mean(rec_scores), 2)
    recall_std = np.round(np.std(rec_scores), 2)

    logging.info("Mean accuracy: {}±{}".format(accuracy_mean, accuracy_std))
    logging.info("Mean precision: {}±{}".format(precision_mean, precision_std))
    logging.info("Mean recall: {}±{}".format(recall_mean, recall_std))
    logging.info("Mean F1: {}±{}".format(f1_mean, f1_std))

    # save metrics to file
    with open(
        f"{args.save_dir}/classification_report_2d_{args.model}.txt",
        "w",
    ) as f:
        f.write(f"Accuracy mean: {accuracy_mean}±{accuracy_std}\n")
        f.write(f"Precision mean: {precision_mean}±{precision_std}\n")
        f.write(f"Recall mean: {recall_mean}±{recall_std}\n")
        f.write(f"F1 mean: {f1_mean}±{f1_std}\n")
