class config:
    def __init__(self, 
            model_name,
            seed ,
            batch_size,
            img_size ,
            num_tiles,
            num_classes,
            num_splits ,
            num_epochs ,
            learning_rate,
            num_workers ,
            verbose ,
            train_data_name,
            last_conv_layer_name,
            classifier_layer_names):
        self.train_data_name=train_data_name
        self.backbone_train_path='tiles/{}/'.format(train_data_name)
        self.model_name=model_name
        self.seed = seed
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_tiles = num_tiles
        self.num_classes = num_classes
        self.num_splits = num_splits
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.num_workers = num_workers
        self.verbose = verbose
        self.train_csv = "tiles/{}.csv".format(train_data_name)
        self.last_conv_layer_name=last_conv_layer_name
        self.classifier_layer_names=classifier_layer_names
        #backbone_test_path = '../input/prostate-cancer-grade-assessment/test_images/'

seresnext50_config= config(seed = 2023,
                model_name="seresnext50",
                batch_size = 4,
                img_size = 64,
                num_tiles = 16,
                num_classes = 6,
                num_splits = 5,
                num_epochs = 15,
                learning_rate = 3e-3,
                num_workers = 1,
                verbose = True,
                train_data_name="64_4_4_0",
                last_conv_layer_name = 'activation_80',
                classifier_layer_names = [
                    'global_average_pooling2d_16',
                    'dense'
                    ]
                )
vgg16_config= config(seed = 2023,
                model_name="vgg16",
                batch_size = 4,
                img_size = 64,
                num_tiles = 16,
                num_classes = 6,
                num_splits = 5,
                num_epochs = 15,
                learning_rate = 3e-3,
                num_workers = 1,
                verbose = True,
                train_data_name="64_4_4_0",
                last_conv_layer_name = 'block5_pool',
                classifier_layer_names = [
                    'global_average_pooling2d',
                    'dense',
                    'batch_normalization',
                    'dense_1'
                    ]
                )
densenet121_config= config(seed = 2023,
                model_name="densenet121",
                batch_size = 4,
                img_size = 64,
                num_tiles = 16,
                num_classes = 6,
                num_splits = 5,
                num_epochs = 15,
                learning_rate = 3e-3,
                num_workers = 1,
                verbose = True,
                train_data_name="64_4_4_0",
                last_conv_layer_name = 'relu',
                classifier_layer_names = [
                    'global_average_pooling2d',
                    'dropout',
                    'dense'
                    ]
                )

efficientNetB2_config= config(seed = 2023,
                model_name="efficientNetB2",
                batch_size = 4,
                img_size = 64,
                num_tiles = 16,
                num_classes = 6,
                num_splits = 5,
                num_epochs = 15,
                learning_rate = 3e-3,
                num_workers = 1,
                verbose = True,
                train_data_name="64_4_4_0",
                last_conv_layer_name = 'top_activation',
                classifier_layer_names = [
                    'global_average_pooling2d',
                    'dropout',
                    'dense'
                    ]
                )