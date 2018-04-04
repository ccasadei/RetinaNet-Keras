# RetinaNet-Keras

Refactory del progetto 
https://github.com/fizyr/keras-retinanet

Il file `configRetinaNet.json` contiene la configurazione della rete, delle classi, del training, i dataset, eccetera.

Per eseguire il training, lanciare il file `training.py`.

Per eseguire il test, lanciare il file `test.py`.

Lanciando il file `debug.py` vengono visualizzate le immagini del dataset con le relative box. Questo serve a capire se l'immagine concorre in modo corretto al training della rete oppure no (se restano dei box rossi).

Nella directory `logs` vengono creati i log per **TensorBoard**.