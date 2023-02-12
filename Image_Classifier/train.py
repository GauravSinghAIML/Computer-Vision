from config import Config
from model import Model
from data import Data

class Train:
    def __init__(self):
        pass
    def train(self):
        config = Config().get_config()
        config['img_path_dir'] = 'C:\Git_Proj\Computer-Vision\Image_Classifier\Image_data'
        config['val_split'] = .2
        config['subset'] = 'both'
        config['seed'] = 123
        config['imag_height'] = 180
        config['img_width'] = 180
        config['batch_size'] = 32
        config['device'] = 'CPU'
        config['epochs'] = 12
        config['model_save_path'] = 'C:\Git_Proj\Computer-Vision\Image_Classifier\Trained_Models\my_model.h5'
        config['training_path'] = 'C:\Git_Proj\Computer-Vision\Image_Classifier\Trained_Models\l_training.png'
        train_ds, val_ds, classes = Data(config).get_dataset()
        obj_train=Model(config)
        model = obj_train.createCustomModel(len(classes))
        history = obj_train.compileAndFitModel(model,train_ds,val_ds)
        obj_train.visualizeTrainingModel(history)
if __name__ == '__main__':
    Train().train()