import os
import shutil
import random
import pathlib
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from sklearn import multiclass
from tensorflow.keras.optimizers import SGD
from keras.metrics import TruePositives
from keras.metrics import TrueNegatives
from keras.metrics import FalseNegatives
from keras.metrics import FalsePositives
import logging
from datetime import date
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras.callbacks import LambdaCallback
from keras.layers import Softmax
from keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model 
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef

def Construct_Testing_Set(output_dir, im_dir, class_labels, class_size):
    os.makedirs(output_dir)
    all_images = list()
    for i in class_labels:
        os.makedirs(os.path.join(output_dir, i))	
        source_dir = os.path.join(im_dir, i)
        class_images = random.sample(os.listdir(source_dir), k = class_size)
        for k in class_images :
            all_images.append(k)
        for j in class_images:
            src = os.path.join(source_dir, j)
            dst = os.path.join(output_dir, i, j)
            shutil.copyfile(src, dst)                  
    output = open(os.path.join(output_dir, output_dir + '_contents.txt'), 'a') 
    output.write(str(all_images))
    output.close()

def Construct_Model_Fitting_Set(output_dir, im_dir, class_labels, test_set_log,class_size, val_ratio = 0):
    # os.makedirs(os.path.join(output_dir, 'model_fit'))
    # os.makedirs(os.path.join(output_dir, 'validation'))
    with open(test_set_log) as f:
        test_set = f.readlines()
    f.close()
    test_set = set(test_set)
    all_images = list()
    for i in class_labels:
        os.makedirs(os.path.join(output_dir, 'model_fit', i))
        os.makedirs(os.path.join(output_dir, 'validation', i))		
        source_dir = os.path.join(im_dir, i)
        class_images = set()
        while len(class_images) < class_size:
            im = random.sample(range(len(os.listdir(source_dir))),k = 1)
            image = os.listdir(source_dir)[im[0]]
            if class_images.isdisjoint(set(image)) & test_set.isdisjoint(set(image)):
                class_images.add(image)           
        for k in class_images :
            all_images.append(k)
        for j in class_images:
            src = os.path.join(source_dir, j)
            val = random.random()
            if val < val_ratio:
                dst = os.path.join(output_dir, 'validation', i, j)
            else: 
                dst = os.path.join(output_dir,'model_fit', i, j)    
            shutil.copyfile(src, dst)           
    output = open(os.path.join(output_dir, output_dir + '_contents.txt'), 'a') 
    output.write(str(all_images))
    output.close()

def Construct_Keras_Directories(output_dir, im_dir, class_labels, class_size = 'max', test_split = .1):
    for i in class_labels:
        print(i)
        source_dir = os.path.join(im_dir, i)
        #Check to make sure dir isnt empty
        initial_count = 0
        for path in pathlib.Path('/home/jharvie/jharvie_project/Datasets/Full_Keyence/'+i).iterdir():
            if path.is_file():
                initial_count += 1
        if initial_count == 0: continue        
        os.makedirs(os.path.join(output_dir, 'Testing', i))	
        os.makedirs(os.path.join(output_dir, 'ModelFitting', i))
        #Need to adjust this so that there can be uneven amount of images per class
        if class_size == 'max':
            sample_size = len(os.listdir(os.path.join(im_dir, i)))
            print(os.path.join(im_dir, i))
            print(class_size)
        else:
            sample_size = class_size       
        class_image_list = random.sample(os.listdir(source_dir), k = sample_size)
        print(len(class_image_list))
        v1 = float(sample_size)
        v2 = test_split
        k_value = round((v1*v2))
        test_image_list = random.sample(class_image_list, k = k_value)
        model_image_list = [x for x in class_image_list if x not in test_image_list]
        class_image_list.clear()
        for im in test_image_list:
            src = os.path.join(source_dir, im)
            dst = os.path.join(output_dir, 'Testing', i, im)
            cmd = f'cp -r {src} {dst}'
            os.system(cmd)
        output = open(os.path.join(output_dir, output_dir + '_Testing_contents.txt'), 'a') 
        output.write(str(test_image_list))
        output.close()
        for im in model_image_list:
            src = os.path.join(source_dir, im)
            dst = os.path.join(output_dir, 'ModelFitting', i, im)
            cmd = f'cp -r {src} {dst}'
            os.system(cmd)    
        output = open(os.path.join(output_dir, output_dir + '_ModelFitting_contents.txt'), 'a') 
        output.write(str(model_image_list))
        output.close()

def Construct_Validation_Splits(im_dir, class_labels, validation_split = 0.2):
    os.makedirs(os.path.join(im_dir, 'Training'), exist_ok = True)
    os.makedirs(os.path.join(im_dir, 'Validation'), exist_ok = True)
    for i in class_labels:
        os.makedirs(os.path.join(im_dir, 'Validation', i), exist_ok = True)
        #check total number of images
        im_num = len(os.listdir(os.path.join(im_dir, i))) 
        val_num = round(im_num*validation_split)
        val_image_list = random.sample(os.listdir(os.path.join(im_dir, i)), k = val_num)
        for im in val_image_list:
            src = os.path.join(im_dir,i,im)
            dst = os.path.join(im_dir, 'Validation', i)
            cmd = f'mv {src} {dst}'
            os.system(cmd)
        src = os.path.join(im_dir,i)
        dst = os.path.join(im_dir, 'Training/')
        cmd = f'mv {src} {dst}'
        os.system(cmd)

def Check_Uniqueness(test_log, model_log):
    with open(test_log) as f:
        test_set = f.readlines()
    f.close()
    test_set = set(test_set)
    with open(model_log) as f:
        model_set = f.readlines()
    f.close()
    model_set = set(model_set)
    check = test_set.isdisjoint(model_set)
    print(check)
    return(check)

def Define_Binary_Model_5Block(a="a"):
	print(a)
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = SGD(lr=0.001, momentum=0.9)
	#Last 4 metrics must be removed when no longer binary 
	model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy','mse', TruePositives(),TrueNegatives(),FalsePositives(),FalseNegatives()])
	return model

def Define_Multi_Model_5Block(num_classes, Input_Shape = (200,200,3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=Input_Shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    #model.add(Dense(num_classes))
    model.add(Dense(num_classes, activation = 'softmax'))
    #model.add(Softmax()) 
	# compile model
	#model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy','mse'])
    model.compile(optimizer='adam', loss = "categorical_crossentropy", metrics=['accuracy'])
    return model

def VGG_ArchitectureA(num_classes, Input_Shape = (224,224,3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=Input_Shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss = "categorical_crossentropy", metrics=['accuracy','mse'])
    return model

def VGG_ArchitectureA_Bin(num_classes, Input_Shape = (224,224,3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=Input_Shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(num_classes, activation = 'sigmoid'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss = "binary_crossentropy", metrics=['accuracy','mse'])
    return model

def VGG_ArchitectureB(num_classes, Input_Shape = (224,224,3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=Input_Shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss = "categorical_crossentropy", metrics=['accuracy','mse'])
    return model

def VGG_ArchitectureB_Bin(num_classes, Input_Shape = (224,224,3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=Input_Shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(num_classes, activation = 'sigmoid'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss = "binary_crossentropy", metrics=['accuracy','mse'])
    return model

def VGG_ArchitectureC(num_classes, Input_Shape = (224,224,3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=Input_Shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss = "categorical_crossentropy", metrics=['accuracy','mse'])
    return model

def VGG_ArchitectureC_Bin(num_classes, Input_Shape = (224,224,3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=Input_Shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_classes, activation = 'sigmoid'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss = "binary_crossentropy", metrics=['accuracy','mse'])
    return model

def VGG_ArchitectureD(num_classes, Input_Shape = (224,224,3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=Input_Shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss = "categorical_crossentropy", metrics=['accuracy','mse'])
    return model

def VGG_ArchitectureD_Bin(num_classes, Input_Shape = (224,224,3)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=Input_Shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation = 'sigmoid'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss = "binary_crossentropy", metrics=['accuracy','mse'])
    return model

def Model_Fit_Binary(model, train_images, val_images, epochs = 1, batch_size = 32, 
    im_size = (200,200), CSVlogfile = "output_epoch_results.csv", save_model = 'NA',log = True):
    if log:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        today = date.today()
        file_handler = logging.FileHandler('keras_functions_' + str(today) + '.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    #This function will only fit and save model as object and/or as binary file4gh
		# create data generator
    datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
    train_im = datagen.flow_from_directory(train_images,
		class_mode='binary' , batch_size=batch_size, target_size=im_size)
    val_im = datagen.flow_from_directory(val_images,
		class_mode='binary' , batch_size=batch_size, target_size=im_size)
    csv_logger = CSVLogger(CSVlogfile)
    progress_logging = LambdaCallback(
		on_epoch_begin=lambda epoch,logs: logger.info('Epoch: ' + str(epoch+1)),
		on_batch_begin=lambda batch,logs: logger.info('Batch: ' + str(batch + 1)))
	# this may need to not save to a new variable but rather return model at end
    model.fit(train_im, epochs = epochs, validation_data = val_im, 
		callbacks=[csv_logger,progress_logging])
    if save_model != 'NA':
        model.save(save_model)
    return model

def Model_Fit_Multi(model, train_images, val_images, epochs = 1, batch_size = 32, 
    im_size = (200,200), weights=None, CSVlogfile = "output_epoch_results.csv", save_model = 'NA',log = True):
    if log:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
        today = date.today()
        file_handler = logging.FileHandler('keras_functions_' + str(today) + '.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
    train_im = datagen.flow_from_directory(train_images,
		class_mode='categorical' , batch_size=batch_size, target_size=im_size)
    val_im = datagen.flow_from_directory(val_images,
		class_mode='categorical' , batch_size=batch_size, target_size=im_size)
    csv_logger = CSVLogger(CSVlogfile)
    progress_logging = LambdaCallback(
		on_epoch_begin=lambda epoch,logs: logger.info('Epoch: ' + str(epoch+1)),
		on_batch_begin=lambda batch,logs: logger.info('Batch: ' + str(batch + 1)))
    model.fit(train_im, epochs = epochs, validation_data = val_im, class_weight = weights,
		callbacks=[csv_logger,progress_logging])
    if save_model != 'NA':
        model.save(save_model)
    return model

def Predict_Model_Multi(test_images, saved_model = False, model_name = 'model', batch_size = 32, im_size = (200,200)):
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_im = datagen.flow_from_directory(test_images, 
        class_mode='categorical', batch_size=batch_size, target_size=im_size, shuffle = False)
    if saved_model: model = load_model(model_name)
    else: model = model_name
    y_predict = model.predict(test_im, verbose = 1)
    y_predict2 = np.argmax(y_predict, axis = 1)
    y_true = test_im.classes
    y_labels = test_im.class_indices
    return y_predict2, y_true, y_labels 

def Predict_Model_Binary(test_images, saved_model = False, model_name = 'model', batch_size = 32, im_size = (200,200)):
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_im = datagen.flow_from_directory(test_images, 
        class_mode='binary', batch_size=batch_size, target_size=im_size, shuffle = False)
    if saved_model: model = load_model(model_name)
    else: model = model_name
    y_predict = model.predict(test_im, verbose = 1)
    y_predict2 = np.argmax(y_predict, axis = 1)
   # y_predict2 = (y_predict > 0.5).astype("int32")
    y_true = test_im.classes
    y_labels = test_im.class_indices
    return y_predict2

def History_Plots_V1(callback_history, name):	
	history = pd.read_csv(callback_history)
	pyplot.plot(1,2,1)
	pyplot.ylim(0, 1)
	pyplot.title('Accuracy')
	pyplot.plot(history.loc[:,"accuracy"], color='blue', label='train')
	pyplot.plot(history.loc[:,'val_accuracy'], color='orange', label='val')
	pyplot.legend() 
	pyplot.tight_layout()
	pyplot.savefig(name + "_acc.png")
	pyplot.close() 

def Confusion_Matrix_Binary(y_true, y_predict, class_labels, name):
    y_pred = (y_predict > 0.5).astype("int32")
    print('Confusion Matrix')
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=class_labels)
    disp.plot()
    pyplot.show()
    pyplot.savefig(name + "_confusion.png")
    pyplot.close() 

def Confusion_Matrix_Multi(y_true, y_predict, class_labels, name):
    y_pred = np.argmax(y_predict, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm,
                     index = class_labels, 
                     columns = class_labels)
    pyplot.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, fmt='g')
    pyplot.title('Confusion Matrix')
    pyplot.ylabel('Actual Values')
    pyplot.xlabel('Predicted Values')
    pyplot.show()                     
    pyplot.savefig(name + "_confusion.png")
    pyplot.close() 

def Precision_Recall_Curve(y_true, y_predict, class_labels, name, average_type = 'micro'):
    if type(class_labels) is dict:
        class_list = list(class_labels.keys())
    elif type(class_labels) is list:
        class_list = class_labels
    else:
        print("check type for class_labels")
    y_pred = np.argmax(y_predict, axis=1)
    f1 = round(f1_score(y_true,y_pred, average= 'micro'),4)    
    y_true = label_binarize(y_true, classes=np.arange(len(class_list)))
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(len(class_list)):
        precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_predict[:, i])
        average_precision[i] = average_precision_score(y_true[:, i], y_predict[:, i])
    precision[average_type], recall[average_type], _ = precision_recall_curve(y_true.ravel(), y_predict.ravel())
    average_precision[average_type] = average_precision_score(y_true, y_predict, average=average_type)    

    colors = sns.color_palette(None, len(class_list)+1)
    _, ax = plt.subplots(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines, labels = [], [] 
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
        plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    display = PrecisionRecallDisplay(
        recall=recall[average_type],
        precision=precision[average_type],
        average_precision=average_precision[average_type])
    display.plot(ax=ax, name=f"{average_type} average precision-recall", color=colors[-1])    
    for i, color in zip(range(len(class_list)), colors):
        display = PrecisionRecallDisplay(
            recall=recall[i],
            precision=precision[i],
            average_precision=average_precision[i])
        class_name = class_list[i]
        display.plot(ax=ax, name=f"Precision-recall for class {class_name}", color=color)
    handles, labels = display.ax_.get_legend_handles_labels()
    handles.extend([l])
    labels.extend(["iso-f1 curves"])    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles=handles, labels=labels, loc="best")
    ax.set_title(f"Class Specific Precision-Recall curve, Global F1 score: {f1}")
    plt.savefig(name + '_Prec-Recall_curves.png')
    plt.close()
    return f1

def Export_Predictions(predict_obj, name):
    d = np.column_stack((predict_obj[0],predict_obj[1]))
    np.savetxt(name + '_PredictOut.txt', d)
    data_labels = predict_obj[2]
    f = open(name + "_Labels.txt","w")
    f.write(str(data_labels))
    f.close()

def Metric_Gen(Pred_df,name):
    f = open(name + "_metrics.txt","w")
    # Global score
    ps_global = precision_score(Pred_df[1],Pred_df[0],average='macro')
    f.write("Prec Global")
    f.write("\n")
    f.write(str(ps_global))
    f.write("\n")
    # Class score 
    ps_class = precision_score(Pred_df[1],Pred_df[0],average=None)
    f.write("Prec Class")
    f.write("\n")
    f.write(str(ps_class))
    f.write("\n")
    #Global recall 
    rs_global = recall_score(Pred_df[1],Pred_df[0],average='macro')
    f.write("Recall Global")
    f.write("\n")
    f.write(str(rs_global))
    f.write("\n")
    # Class recall
    rs_class = recall_score(Pred_df[1],Pred_df[0],average=None)
    f.write("Recall Class")
    f.write("\n")
    f.write(str(rs_class))
    f.write("\n")
    #Global F1 Macro
    F1_global = f1_score(Pred_df[1],Pred_df[0],average='macro')
    f.write("F1 Global Macro")
    f.write("\n")
    f.write(str(F1_global))
    f.write("\n")
    #Global F1 Weight
    F1_global = f1_score(Pred_df[1],Pred_df[0],average='weighted')
    f.write("F1 Global Weight")
    f.write("\n")
    f.write(str(F1_global))
    f.write("\n")
    # Class F1
    F1_class = f1_score(Pred_df[1],Pred_df[0],average=None)
    f.write("F1 Class")
    f.write("\n")
    f.write(str(F1_class))
    f.write("\n")
    #Matthews
    MatCo = matthews_corrcoef(Pred_df[1],Pred_df[0])
    f.write("Matthew's")
    f.write("\n")
    f.write(str(MatCo))
    f.close()
