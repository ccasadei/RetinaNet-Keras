import os
from math import ceil

from keras.utils import plot_model

from config import Config
from model.callbacks import get_callbacks
from model.generator import get_generators
from model.loss import getLoss
from model.optimizer import get_optimizer
from model.resnet import resnet50_retinanet, resnet152_retinanet

# leggo la configurazione
config = Config('configRetinaNet.json')

# creazione del modello
if config.type == '50':
    model, bodyLayers = resnet50_retinanet(len(config.classes), weights='imagenet', nms=True)
elif config.type == '152':
    model, bodyLayers = resnet152_retinanet(len(config.classes), weights='imagenet', nms=True)
else:
    print("Tipo modello non riconosciuto ({})".format(config.type))
    exit(1)

model.summary()

# verifico se esistono dei pesi pre-training
if os.path.isfile(config.pretrained_weights_path):
    model.load_weights(config.pretrained_weights_path, by_name=True, skip_mismatch=True)
    print("Caricati pesi PRETRAINED")
else:
    # altrimenti carico i pesi di base (escludendo i layer successivi a quello indicato, compreso)
    if config.type == '50':
        base_weights = config.base_weights50_path
    elif config.type == '152':
        base_weights = config.base_weights152_path
    if os.path.isfile(base_weights):
        model.load_weights(base_weights, by_name=True, skip_mismatch=True)
        print("Caricati pesi BASE")
    else:
        print("Senza pesi")

if config.type == '50':
    freeze_layer_stop_name = config.freeze_layer_stop_name_50
elif config.type == '152':
    freeze_layer_stop_name = config.freeze_layer_stop_name_152

# eseguo il freeze dei layer più profondi (in base ad una configurazione mi posso fermare)
if config.do_freeze_layers:
    conta = 0
    for l in bodyLayers:
        if l.name == freeze_layer_stop_name:
            break
        l.trainable = False
        conta += 1
    print("")
    print("Eseguito freeze di " + str(conta) + " layers")
    print("Nuovo summary dopo FREEZE")
    print("")
    model.summary()

# compilo il model con loss function e ottimizzatore
model.compile(loss=getLoss(), optimizer=get_optimizer(config.base_lr), metrics=['accuracy'])

if config.model_image:
    plot_model(model, to_file='model_image.jpg')

# preparo i generatori di immagini per il training e la valutazione
train_generator, val_generator, n_train_samples, n_val_samples = get_generators(config.images_path,
                                                                                config.annotations_path,
                                                                                config.train_val_split,
                                                                                config.batch_size,
                                                                                config.classes,
                                                                                transform=config.augmentation,
                                                                                debug=False)

# preparo i callback
callbacks = get_callbacks(config)

# eseguo il training
model.fit_generator(generator=train_generator,
                    steps_per_epoch=ceil(n_train_samples / config.batch_size),
                    epochs=config.epochs,
                    callbacks=callbacks,
                    validation_data=val_generator,
                    validation_steps=ceil(n_val_samples / config.batch_size))

# salvo i pesi
model.save_weights(config.trained_weights_path)
