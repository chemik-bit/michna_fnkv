import tensorflow as tf
def contrastive_loss(y_true, y_pred):
    """Calculates the constrastive loss.
    Arguments:
        y_true: List of labels, each label is of type float32.
        y_pred: List of predictions of same length as of y_true,
                each label is of type float32.
    Returns:
        A tensor containing constrastive loss as floating point value.
    """
    margin=1
    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        (1 - y_true) * square_pred + (y_true) * margin_square
    )