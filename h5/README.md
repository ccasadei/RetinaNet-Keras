I pesi di **BASE** sono scaricabili da qui: 

https://github.com/fizyr/keras-models/releases

Sono disponibili vari pesi a seconda del modello ResNet applicato (50, 101, 152)

Il peso va rinominato in `base.h5`.

Il file `pretrained.h5` se esiste è quello che viene caricato all'inizio del training come partenza (è impostato sulle classi indicate in `configRetinaNet.json`).

Il file `result.h5` se esiste è quello creato alla fine dell'allenamento.

Il file `chkpnt_best.h5` se esiste è quello creato durante l'allenamento come checkpoint migliore. Può essere rinominato in `pretrained.h5` se si interrompe l'allenamento e successivamente si vuole ripartire da quel punto.

Tutti i nomi ed i path dei file sono comunque configurabili da `configRetinaNet.json`.


