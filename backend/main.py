# Symptom match code
# Predict disease using symptoms 
# 1) Decision Tree or Random Forest
# 2) Neural Networks with softmax output layer -- for a rank of diseases depending on probability of disease match

# import libraries
from xml.etree.ElementPath import prepare_descendant
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense

features = np.array(['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea','mild_fever',
        'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
        'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes',
        'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
        'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion',
        'chest_pain', 'weakness_in_limbs', 'fast_heart_rate',
        'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
        'irritation_in_anus', 'neck_pain', 'dizziness','cramps', 'bruising',
        'obesity','swollen_legs', 'swollen_blood_vessels',
        'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
        'swollen_extremeties', 'excessive_hunger',
        'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech',
        'knee_pain','hip_joint_pain', 'muscle_weakness', 'stiff_neck',
        'swelling_joints', 'movement_stiffness', 'spinning_movements',
        'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
        'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
        'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching',
        'toxic_look_(typhos)', 'depression', 'irritability', 'muscle_pain',
        'altered_sensorium', 'red_spots_over_body', 'belly_pain',
        'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes',
        'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
        'rusty_sputum', 'lack_of_concentration', 'visual_disturbances',
        'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma',
        'stomach_bleeding', 'distention_of_abdomen',
        'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum',
        'prominent_veins_on_calf', 'palpitations', 'painful_walking',
        'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
        'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails',
        'blister', 'red_sore_around_nose', 'yellow_crust_ooze'])

# list of all included diseases
diseases = np.array(['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis',
        'Drug Reaction', 'Peptic ulcer diseae', 'AIDS', 'Diabetes',
        'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',  'Migraine',
        'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice',
        'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
        'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
        'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia',
        'Dimorphic hemmorhoids(piles)', 'Heart attack','Varicose veins',
        'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia',
        'Osteoarthristis', 'Arthritis',
        '(vertigo) Paroymsal  Positional Vertigo','Acne',
        'Urinary tract infection','Psoriasis','Impetigo'])


def DecisionTree(X_train, y_train, X_test, y_test):
    # 96% accuracy

    #train and fit model
    model = tree.DecisionTreeClassifier()
    model.fit(X_train, y_train)

    #test model on test set
    predictions = model.predict(X_test)

    wrong = 0
    total = len(y_test)
    for i in range(total):
        if predictions[i]!=y_test[i]:
            wrong+=1
            # print(predictions[i]," -- ", y_test[i])
    
    print("Accuracy Decision Tree: {}/{} are corrrect -- {}%".format(total-wrong, total, (total-wrong)/total*100))
    return


def RandomForest(X_train, y_train, X_test, y_test):
    # 100% accuracy    

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    wrong = 0
    total = len(y_test)
    for i in range(total):
        if predictions[i]!=y_test[i]:
            wrong+=1
            # print(predictions[i]," -- ", y_test[i])
    
    print("Accuracy RandomForest: {}/{} are corrrect -- {}%".format(total-wrong, total, (total-wrong)/total*100))
    return

def NeuralNet(X_train, y_train, X_test, y_test, n=5):

    X_train=tf.strings.to_number(X_train, out_type=tf.dtypes.int64)
    X_test=tf.strings.to_number(X_test, out_type=tf.dtypes.int64)

    #convert y_train to numerical
    y_train_num = []
    for dis in y_train:
        #TODO: make dis case insensitive
        dis = dis.strip()
        y_train_num.append(np.where(diseases==dis))
    y_train_num = np.array(y_train_num)
    # print(y_train_num)

    for i in range(len(y_train)):    
        assert y_train[i].strip()==diseases[y_train_num[i]][0][0].strip()
        # print(y_train[i], y_train_num[i], diseases[y_train_num[i]])

    model = Sequential(
    [ 
        Dense(400, activation = 'relu'),
        Dense(25, activation = 'relu'),
        Dense(120, activation = 'relu'),
        Dense(15, activation = 'relu'),
        Dense(len(diseases), activation = 'linear')    # < softmax activation here
    ]
    )

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(0.001),
    )

    model.fit(
        X_train,y_train_num,
        epochs=10
    )

    predictions = model.predict(X_test)

    #convert to probabilities (of each disease)
    prob_disease = tf.nn.softmax(predictions).numpy()
    prediction_names = []

    #convert probabilities to categorical values == disease names
    #by choosing highest prob index as predicted disease

    for prob in prob_disease:
        i = prob.argsort()[-n:][::-1]
        i = np.array(i)
        prediction_names.append(diseases[i])

    
    wrong = 0
    total = len(y_test)
    for i in range(total):
        if prediction_names[i][0]!=y_test[i]:
            wrong+=1
            # print(predictions[i]," -- ", y_test[i])
    
    print("Accuracy Nueral Network: {}/{} are corrrect -- {}%".format(total-wrong, total, (total-wrong)/total*100))


    return

# load data
def load_data():
    # list of features == symptoms in this case

    data = csv_to_data()
    X , y = [x[:-1] for x in data], [x[-1] for x in data]
    X = np.asarray(X)
    y = np.asarray(y)
    
    return X, y

def csv_to_data():
    train = []
    with open("symptoms_data.csv", mode="r") as file:

        #read csv file
        csvfile = csv.reader(file)
        next(csvfile)
        for lines in csvfile:
            train.append([int(x) for x in lines[:-1]]+[lines[-1]])
    
    train = np.asarray(train)
    return train


def main():
    
    # print(load_data())

    X, y = load_data()
    # split data into train, test and validation sets
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.8)
    X_test, X_val, y_test, y_val = train_test_split(X_rem, y_rem, test_size=0.5)

    # DecisionTree(X_train, y_train, X_rem, y_rem)
    # RandomForest(X_train, y_train, X_rem, y_rem)
    NeuralNet(X_train, y_train, X_rem, y_rem)
    return

if __name__ == "__main__":
    main()


# split data into train, test, and validation sets
# train data
    # decision tree
    # random forest
    # neural networks   

# test models / inference
