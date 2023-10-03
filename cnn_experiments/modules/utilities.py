import tensorflow as tf


def generate_test_batch(batch_size, image_width, image_height, n_channels=3):
    """
    """
    return tf.cast(
        tf.random.uniform(
            (batch_size, image_width, image_height, n_channels),
            minval=0,
            maxval=128,
            dtype=tf.int32
        ),
        dtype=tf.float32
    )
