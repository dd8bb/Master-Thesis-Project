# coding: utf-8
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from dataset import Dataset
import math
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def build_model(input_shape):
    
    #Create model
    model = keras.Sequential([
        #1st 2DConv Layer
        layers.Conv2D(16, (8, 6), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Activation('relu'),
       
        #MaxPooling
        layers.MaxPool2D((3,6), padding='same'),

        #2nd 2Dconv Layer
        layers.Conv2D(32, (6,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        
        #MaxPooling
        #layers.MaxPool2D((1,3), padding='same'),
        
        #Flatten the output and feed it into dense layer
        layers.Flatten(),
        layers.Dropout(0.25), #Add dropout to FC layer to avoid overfitting
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), 
        #Output Layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    return model
# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.2
    drop = 0.85
    epochs_drop = 1.0
    
    if epoch > 14:
        lrate = 0.02
    else:
        lrate = 0.05
    #lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate
def exp_decay(epoch):
    initial_lrate = 0.1
    k = 0.1
    lrate = initial_lrate * math.exp(-k*epoch)
    return lrate
# Momentum schedule

        
# Save model weights
PATH_TO_SAVED_MODELS = "../Resultados/models"

def save_model(model, name):
    
    path = os.path.join(PATH_TO_SAVED_MODELS, name + ".h5")
    model.save(path)
    print("Saved model to disk")

def load_model(name):
    
    path = os.path.join(PATH_TO_SAVED_MODELS, name)
    model = keras.models.load_model(path)
    print("Model succesfully loaded")
    return model
    
# f-score evaluation
fps_dict = {
            "Low" : 3.59,
            "Mid" : 7.18,
            "High": 14.35,
            }

def find_matches(label, predictions, tol):
    posible_match = []
    for pred in predictions : # For each label evaluate the distance between all predicts
        if abs(label-pred) < tol/2:
            posible_match.append(pred) # predicts within tolerance boundary are possible matches
    return posible_match

def find_min_dist(label,predictions, posible_match_list):
    selected_pred_index = None
    min = 100
    #We find the closest match for this label
    for pred in posible_match_list:
        if abs(label-pred) < min:
            min = abs(label-pred)
            selected_pred_index = np.where(predictions == pred) #Get index for selected prediction
    return selected_pred_index, min


def cal_f_score(labels, predictions, tol, res):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    tol_frames = fps_dict[res] * tol
    while len(labels) > 0:
        label = labels[0]
        label_index = np.where(labels == label) # Get index for selected label
        posible_match = find_matches(label, predictions, tol_frames)
        
        if len(posible_match) == 0: #If there is no match we drop the label and continue
            labels = np.delete(labels, label_index)
            false_negatives += 1 # Model predict this label with a 0 thus is a false negative.
        else:
            # First we get the closest match
            pred_index_label_1, dmin_label_1 = find_min_dist(label, predictions, posible_match)
           
            # Now we have to check if there is another label wich is more closer to the pred than the first one
            if len(labels) > 1:
                label_2 = labels[1]
                if abs(label_2 - label) < tol_frames: #If its under the tolerance gap there are chances evaluated pred could be for one of these labels
                    posible_match_2 = find_matches(label_2, predictions, tol_frames)
                    pred_index_label_2, dmin_label_2 = find_min_dist(label_2, predictions, posible_match_2)
            
                    if pred_index_label_1 == pred_index_label_2: #If both share the same prediction as the closest one 
                        #Then we have to compare its distances between label and predict
                        if dmin_label_1 <= dmin_label_2:
                            #First label is the predicted one
                            predictions = np.delete(predictions, pred_index_label_1)
                            labels = np.delete(labels, label_index)
                            true_positives += 1
                        else:
                            #Second label is the predicted one, so first label is a false negative
                            labels = np.delete(labels, label_index)
                            false_negatives += 1
                    else:
                        #If each label have its closest pred, then the pred for the first label is OK.
                        predictions = np.delete(predictions, pred_index_label_1)
                        labels = np.delete(labels, label_index)
                        true_positives += 1
                
                else:
                    #If there is no closest labels then we remove this prediction and label 
                    predictions = np.delete(predictions, pred_index_label_1)
                    labels = np.delete(labels, label_index)
                    true_positives += 1 # Model predict this label correctly so is a true positive
            
            else:#Last label
                predictions = np.delete(predictions, pred_index_label_1)
                labels = np.delete(labels, label_index)
                true_positives += 1 # Model predict this label correctly so is a true positive
                
    #end while loop
    false_positives = len(predictions) #All remain predictions are target as false positives
    
    #Calculate recall, precision and f-score:
    recall = true_positives / (true_positives+false_negatives)
    if true_positives == 0 and false_positives == 0:
        precision = 0
    else:
        precision = true_positives / (true_positives+false_positives)
    f_score = true_positives / (true_positives + 0.5*(false_positives+false_negatives))
    
    return f_score, recall, precision
            
        
            
def bagging(test_ds, n_models, res, context, smear, extra=None):
    
    
    
    if str(smear).find("."):
        model_name = res.capitalize() + "_" + str(context) + "_" + str(smear).replace(".","p")
    else:
        model_name = res.capitalize() + "_" + str(context) + "_" + str(smear)
    
    if extra != None:
        model_name = model_name + "_" + extra
    
    for i in range(1,n_models+1):
        sufix = model_name  + "_" + str(i) + ".h5"
        model = load_model(sufix)
        test_ds.predict(model)
        del model
        if i == 1:
            predictions = [test_ds.pred_labels]
        else:
            predictions.append(test_ds.pred_labels)
            
        stacked_preds = np.stack(predictions, axis=0)
    
    return np.mean(stacked_preds,axis=0)
        

def threshold_optimization(test_ds, tol=3, distance=7, res = "Mid"):
    f_scores = []
    recalls = []
    precisions = []
    heights = []
    y_labels_index = np.where(test_ds.oY_data > 0)[0]

    for i in range(1,40):
        height = i/40
        heights.append(height)
        peaks, _ = find_peaks(test_ds.pred_labels[:,0], height=height, distance = distance)
        f_score, recall, precision = cal_f_score(y_labels_index, peaks, tol=tol, res=res)
        f_scores.append(f_score)
        recalls.append(recall)
        precisions.append(precision)
    
    plt.figure(figsize=(25,10))
    f, =plt.plot(f_scores)
    f.set_label('F-Score')
    f.set_linestyle('dashdot')
    r, = plt.plot(recalls)
    r.set_label('Recall')
    r.set_linestyle('dotted')
    p, = plt.plot(precisions)
    p.set_label('Precision')
    p.set_linestyle('dashed')
    plt.legend(fontsize='x-large')
    plt.xlabel("Peak Threshold")
    plt.ylabel("Percentage")
    plt.title("Threshold Optimization")
    hi =list(range(len(heights)))
    a = plt.xticks(hi, heights)
    
    
    max_f_score = max(f_scores)
    indexes = []

    for i in range(len(f_scores)):
        if f_scores[i] == max_f_score:
            indexes.append(i)

    #best recall with best f_score
    max_recall = 0
    for index in indexes:
        if recalls[index] > max_recall:
            max_recall = recalls[index]
            opt_height = index
            
    opt = plt.axvline(x=opt_height, color="r")
    opt.set_label('Optimum Height')
    plt.show()
    print(f"Optimum Height set at {heights[opt_height]}")
    
        
        
