import json

class Config:

    def __init__(self, config_path):
        with open(config_path) as config_buffer:
            config = json.loads(config_buffer.read())

        self.batch_size = config['train']['batch_size']
        self.epochs = config['train']['epochs']
        self.base_lr = config['train']['base_lr']
        self.patience = config['train']['patience']
        self.do_freeze_layers = config['train']['do_freeze_layers']
        self.freeze_layer_stop_name_50 = config['train']['freeze_layer_stop_name_50']
        self.freeze_layer_stop_name_152 = config['train']['freeze_layer_stop_name_152']
        self.train_val_split = config['train']['train_val_split']
        self.augmentation = config['train']['augmentation']

        self.pretrained_weights_path = config['path']['pretrained_weights']
        self.base_weights50_path = config['path']['base_weights_50']
        self.base_weights152_path = config['path']['base_weights_152']
        self.trained_weights_path = config['path']['trained_weights']
        self.chkpnt_weights_path = config['path']['chkpnt_weights']
        self.images_path = config['path']['images']
        self.annotations_path = config['path']['annotations']
        self.test_images_path = config['path']['test_images']
        self.test_result_path = config['path']['test_result']
        self.log_path = config['path']['log']

        self.type = config['model']['type']
        self.model_image = config['model']['model_image']
        self.classes = config['model']['classes']

        self.test_save_annotations = config['test']['save_annotations']
        self.test_start_index = config['test']['start_index']
