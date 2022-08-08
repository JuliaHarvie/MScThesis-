from main_functions import VGG_ArchitectureA
from main_functions import Model_Fit_Multi
from main_functions import Predict_Model_Multi
from main_functions import Export_Predictions
from main_functions import Metric_Gen
import numpy as np

model = VGG_ArchitectureA(num_classes=6)

Model_Fit_Multi(model,
                train_images = 'ModelFitting/Training',
                val_images = 'ModelFitting/Validation',
                epochs = 15,
                batch_size = 128,
                im_size = (224,224),
                CSVlogfile = "ModelA_DataSet1_v2_epoch_results.csv",
                save_model = 'saved_models/ModelA_DataSet1',
                log = True)

out = Predict_Model_Multi(test_images = "/DataSet1_Testing",
                     saved_model = True,
                     model_name = 'saved_models/ModelA_v2_DataSet1',
                     batch_size = 128,
                     im_size = (224,224))

Export_Predictions(out, 'ModelA_DataSet1')

test_df = np.loadtxt('ModelA_DataSet1_PredictOut.txt',unpack=True)
Metric_Gen(test_df, "ModelA_DataSet1_Final")
