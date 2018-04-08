from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, EarlyStopping


def get_callbacks(config):
    return [
        ModelCheckpoint(config.chkpnt_weights_path,
                        monitor='val_loss',
                        verbose=1,
                        save_best_only=True,
                        save_weights_only=True,
                        mode='auto',
                        period=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                          patience=min(2, config.patience / 10),
                          verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0),
        EarlyStopping(monitor='val_loss',
                      min_delta=0.0001,
                      patience=config.patience),
        TensorBoard(log_dir=config.log_path)
    ]

