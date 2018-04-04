import keras

from model.tensorflow_backend import where, gather_nd


def getLoss():
    return {
        'regression': smooth_l1(),
        'classification': focal()
    }


def focal(alpha=0.25, gamma=2.0):
    def _focal(y_true, y_pred):
        labels = y_true
        classification = y_pred

        # calcola il focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # filtra gli anchor da ignorare
        anchor_state = keras.backend.max(labels, axis=2)  # -1 = ignora, 0 = background, 1 = oggetto
        indices = where(keras.backend.not_equal(anchor_state, -1))
        cls_loss = gather_nd(cls_loss, indices)

        # calcola il normalizzatore come il numero di anchor positivi (-1 = ignora, 0 = background, 1 = oggetto)
        normalizer = where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(1.0, normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # separa target e stato
        regression = y_pred
        regression_target = y_true[:, :, :4]
        anchor_state = y_true[:, :, 4]

        # calcola la "smooth L1 loss"
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    altrimenti
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # filtra gli anchor da ignorare (-1 = ignora, 0 = background, 1 = oggetto)
        indices = where(keras.backend.equal(anchor_state, 1))
        regression_loss = gather_nd(regression_loss, indices)

        # calcola il normalizzatore come il numero di achor positivi (-1 = ignora, 0 = background, 1 = oggetto)
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(keras.backend.maximum(1, normalizer), dtype=keras.backend.floatx())

        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1
