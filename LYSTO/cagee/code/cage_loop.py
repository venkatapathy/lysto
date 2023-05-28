import argparse
import math
import os
import pickle
import enum
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from spear.labeling import PreLabels, labeling_function, ABSTAIN, continuous_scorer, LFSet
from spear.cage import Cage
from utils import custom_dataset, custom_random_dataset, custom_random_seeds, random_seed, train_lf, train_all_LF, binary_model_LF, create_cnn
import warnings
warnings.filterwarnings('ignore')


def binary_lf_factory(path, ClassLabels, cl, other_class, model_name):
    @continuous_scorer()
    def model_clv_ocv(x,**kwargs):
        x = np.array(x).flatten()  
        model = pickle.load(open(path+f'{cl.value}_{other_class.value}_{model_name}.pkl','rb'))
        confidence_scores = model.predict_proba([x])
        return float(confidence_scores[0][1])
    
    @labeling_function(name=f"LF_{model_name}_{cl.value}_{other_class.value}",cont_scorer=model_clv_ocv, label=cl)
    def LF_model_clv_ocv(x, **kwargs):
        x = np.array(x).flatten()  
        model = pickle.load(open(path+f'{cl.value}_{other_class.value}_{model_name}.pkl','rb'))
        if model.predict_proba([x])[0][1]>0.8: 
            return cl
        else: 
            return ABSTAIN
        
    return LF_model_clv_ocv


def lf_factory(path, ClassLabels, cl, model_name):
    @continuous_scorer()
    def model_clv(x,**kwargs):
        x = np.array(x).flatten()  
        model = pickle.load(open(path+f'{cl.value}_{model_name}.pkl','rb'))
        confidence_scores = model.predict_proba([x])
        return float(confidence_scores[0][1]) 
    
    @labeling_function(name=f"LF_{model_name}_{cl.value}",cont_scorer=model_clv, label=cl)
    def LF_model_clv(x, **kwargs):
        x = np.array(x).flatten()  
        model = pickle.load(open(path+f'{cl.value}_{model_name}.pkl','rb'))
        if model.predict_proba([x])[0][1]>0.8: 
            return cl
        else: 
            return ABSTAIN
        
    return LF_model_clv

def createLFs(classes,model_nums,binary_model,path):
    
    # Declaring Class Labels
    ABSTAIN = None
    enum_dict = {"BASOPHIL": 0, "EOSINOPHIL": 1, "ERYTHROBLAST": 2, "IMG": 3, "LYMPHOCYTE": 4, "MONOCYTE": 5, "NEUTROPHIL": 6, "PLATELET": 7}
    enum_keys = list(enum_dict.keys())
    enum_dict = {enum_keys[k]:classes.index(enum_dict[enum_keys[k]]) for k in range(len(enum_keys)) if enum_dict[enum_keys[k]] in classes}
    ClassLabels = enum.Enum('ClassLabels', enum_dict)
    models = ['lr','svm','rf','knn','dtc']
    models = [models[i] for i in range(len(models)) if i in model_nums]
    LFS = []

    # Creating Labelling Functions
    if binary_model:
        for cl in ClassLabels:
            for other_class in ClassLabels:
                if cl.value!=other_class.value:
                    for model_name in models:
                        LFS.append(binary_lf_factory(path, ClassLabels, cl, other_class, model_name))
                        
    else:
        for cl in ClassLabels:
            for model_name in models:
                LFS.append(lf_factory(path, ClassLabels, cl, model_name))

    return LFS, ClassLabels

def map (curr, cls):
    if curr==cls:
        return 1
    else:
        return 0
    
def cnnScore(x, y,dataset, classes):
    x_train = np.array(x).reshape(-1, 28, 28, 3)
    x_train = x_train.astype("float32") / 255
    y_train = [int(i) for i in y]
    y_train = np_utils.to_categorical(y_train, len(classes))

    # Load Validation Data
    x_val, y_val = dataset["val_images"].copy(), dataset["val_labels"].copy()
    x_val = np.array(x_val).reshape(-1, 28, 28, 3)
    x_val = x_val.astype("float32") / 255
    y_val = [int(i) for i in y_val] 

    xv = np.array(x_val)
    yv = np.array(y_val)
    yv = np_utils.to_categorical(yv, num_classes=len(classes))

    batch_size = 128
    epochs = 25
    model = create_cnn(num_classes = len(classes))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, verbose=0)
    return model.evaluate(xv, yv, verbose = 0)


def cage_loop_multiseed(LFS, max_iters, threshold, img_per_class, classes,label_frac,data_path,save_path,ClassLabels, compounding, binary_model):
    # Paths
    log_path_cage = './cage_loop/log.txt' 
    path_json = "./cage_loop/labels.json"
    U_path_pkl = "./cage_loop/unlabelled.pkl"

    # Loading Variables and Data
    print("Classes used in expt:",classes)
    dataset,seeds_x,seeds_y = custom_random_seeds(classes=classes, path=data_path, fraction=label_frac, n_seeds=max_iters)
    xu = np.array(dataset['rem_images'])
    yu = np.array(dataset['rem_labels'])
    labeled_x = seeds_x[0].copy()
    labeled_y = seeds_y[0].copy()
    overallOracle = seeds_y[0].copy()

    # Creating rules
    n_lfs = len(LFS)
    rules = LFSet("BM_LF")
    rules.add_lf_list(LFS)
    
    confidence_list = []
    val_scores = []
    oracle_val_scores = []
    classwise_accuracies = []
    pl_accuracies = []
    baseline = []
    for i in range(max_iters):
        # Load Data
        if compounding:
            x = seeds_x[0].copy()
            y = seeds_y[0].copy()
            for seedi in range(1,i+1):
                x = np.append(x,seeds_x[seedi], axis=0)
                y = np.append(y,seeds_y[seedi], axis=0)
        else:
            x = seeds_x[i]
            y = seeds_y[i]
        labeled_x = np.append(labeled_x,seeds_x[i], axis=0)
        labeled_y = np.append(labeled_y,seeds_y[i], axis=0)
        overallOracle = np.append(overallOracle,seeds_y[i], axis=0)

        # Baseline
        baseline.append(cnnScore(x,y,dataset,classes)[1]*100)

        # Train Models in LFs
        if binary_model:
            binary_model_LF(x,y,len(classes),save_path,label_frac)
        else:
            train_all_LF(x,y,len(classes),save_path,label_frac)

        # Unlabelled
        u_noisy_labels = PreLabels(name="bmnist_rem_ul",
                                    data=xu,
                                    rules=rules,
                                    labels_enum=ClassLabels,
                                    num_classes=len(classes))
        # Lu,Su = u_noisy_labels.get_labels()
        u_noisy_labels.generate_pickle(U_path_pkl)
        u_noisy_labels.generate_json(path_json)

        # Cage
        cage = Cage(path_json = path_json, n_lfs = n_lfs)
        
        probs = cage.fit_and_predict_proba(path_pkl = U_path_pkl, path_log = log_path_cage, qt = 0.9, qc = 0.85, metric_avg = ['macro'], n_epochs = 100, lr = 0.01)
        labels = np.argmax(probs, 1)

        print("="*135)
        print("Iteration",i)
        values, frequency = np.unique(yu, return_counts=True)
        for values, frequency in zip(values, frequency):
            print(f"Labels of Lake Class {values}: {frequency}")

        values, frequency = np.unique(y, return_counts=True)
        for values, frequency in zip(values, frequency):
            print(f"Labels of Labelled Set {values}: {frequency}")
        
        
        
        print("Shape of Labeled Data:",x.shape)
        print("Shape of Unlabeled Data:",xu.shape)
        print("Accuracy on unlabelled images:",accuracy_score(labels,yu)*100)
        
        
        # cage.save_params(save_path = params_path)

        confidence = np.array([np.max(i) for i in probs])
        confidence_list.append(confidence)
        print(i,probs.shape)

        # Getting indices of probabilities in decreasing order
        idx = np.argsort(confidence)
        idx = idx[::-1] 

        # plt.yscale("log")
        # plt.plot(confidence[idx])
        # plt.show()

        # Number of images per class (5%)
        # img_per_class = int(0.05*len(confidence)/len(classes))

        # Number of images per class (50)
        
        
        print("Num img per class =",img_per_class)

        pop_list = [] #list of indices of images to be added
        label_count = []

        for j in idx:
            if confidence[j]>threshold and label_count.count(labels[j])<img_per_class:
                pop_list.append(j)
                label_count.append(labels[j])
        
        print("Number of images getting transferred:", len(pop_list))
        print('Accuracy of Pseudo-labelled img added to dataset:', accuracy_score(labels[pop_list],yu[pop_list])*100)
        pl_accuracies.append(accuracy_score(labels[pop_list],yu[pop_list])*100)

        # Confusion Matrix & Classwise Accuracies
        cmidx = [[(map(yu[j],i), map(labels[j],i)) for j in range(len(labels)) if labels[j]==i] for i in range(len(classes))]
        
        fig, ax = plt.subplots(math.ceil(len(classes)/3),3, figsize=(20, 5*math.ceil(len(classes)/3)))
        pltnum = 0
        cacc = []
        for cmid in cmidx:
            cacc.append(accuracy_score([z[0] for z in cmid], [z[1] for z in cmid])*100)
            ax[pltnum//3,pltnum%3].set_title(f"Class {pltnum}")
            confusion_matrix = metrics.confusion_matrix([z[0] for z in cmid], [z[1] for z in cmid])
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
            try:
                cm_display.plot(ax=ax[pltnum//3,pltnum%3])
            except:
                pass
            pltnum += 1
        classwise_accuracies.append(cacc)
        fig.suptitle(f"Iteration {i}")
        plt.savefig(save_path+f'CM_{i}.png')
        

        if len(pop_list)<50:
            break
        
        x = np.append(x,xu[pop_list], axis=0)
        y = np.append(y,labels[pop_list], axis=0)

        labeled_x = np.append(labeled_x,xu[pop_list], axis=0)
        labeled_y = np.append(labeled_y,labels[pop_list], axis=0)
        overallOracle = np.append(overallOracle,yu[pop_list], axis=0)

        xu = np.delete(xu,pop_list, axis=0)
        yu = np.delete(yu,pop_list, axis=0)

        # Deleting variables
        del u_noisy_labels
        del cage

        val_scores.append(cnnScore(labeled_x, labeled_y,dataset,classes)[1]*100)
        oracle_val_scores.append(cnnScore(labeled_x,overallOracle,dataset,classes)[1]*100)
        print(f"CNN Val accuracy trained on Seed Set for iteration {i}: ", baseline[i])
        print(f"CNN Val accuracy trained on Lake Set for iteration {i}: ", val_scores[i])
        print(f"CNN Val accuracy trained on Oracle for iteration {i}: ", oracle_val_scores[i])
        # print(f"Classwise Precisions for iteration {i}:",classwise_accuracies[i])

        if yu.shape[0]<50:
            break

        
        print("="*135)

    return x,y,xu,yu,confidence_list, val_scores, oracle_val_scores, classwise_accuracies, pl_accuracies, baseline

def cage_loop(LFS, max_iters, threshold, img_per_class, classes,label_frac,data_path,save_path,ClassLabels,random,binary_model):
    # Paths
    log_path_cage = './cage_loop/log.txt'
    path_json = "./cage_loop/labels.json"
    U_path_pkl = "./cage_loop/unlabelled.pkl"

    # Loading Data
    if random:
        dataset,x,y = custom_random_dataset(classes=classes, path=data_path, fraction=label_frac)
    else:
        dataset,x,y = custom_dataset(classes=classes, path=data_path, fraction=label_frac)

    xu = np.array(dataset['rem_images'])
    yu = np.array(dataset['rem_labels'])
    yOracle = copy.deepcopy(y)
    
    print("Classes used in expt:",classes)

    # Creating rules
    n_lfs = len(LFS)
    rules = LFSet("BM_LF")
    rules.add_lf_list(LFS)
    
    confidence_list = []
    val_scores = []
    oracle_val_scores = []
    classwise_accuracies = []
    pl_accuracies = []
    baseline=[cnnScore(x,y,dataset,classes)[1]*100]
    for i in range(max_iters):
        # Train Models in LFs
        if binary_model:
            binary_model_LF(x,y,len(classes),save_path,label_frac)
        else:
            train_all_LF(x,y,len(classes),save_path,label_frac)

        # Unlabelled
        u_noisy_labels = PreLabels(name="bmnist_rem_ul",
                                    data=xu,
                                    rules=rules,
                                    labels_enum=ClassLabels,
                                    num_classes=len(classes))
        u_noisy_labels.generate_pickle(U_path_pkl)
        u_noisy_labels.generate_json(path_json)

        # Cage
        cage = Cage(path_json = path_json, n_lfs = n_lfs)
        
        probs = cage.fit_and_predict_proba(path_pkl = U_path_pkl, path_log = log_path_cage, qt = 0.9, qc = 0.85, metric_avg = ['macro'], n_epochs = 100, lr = 0.01)
        labels = np.argmax(probs, 1)


        #print(labels)

        print("="*135)
        print("Iteration",i)
        values, frequency = np.unique(yu, return_counts=True)
        for values, frequency in zip(values, frequency):
            print(f"Labels of Lake Class {values}: {frequency}")

        values, frequency = np.unique(y, return_counts=True)
        for values, frequency in zip(values, frequency):
            print(f"Labels of Labelled Set {values}: {frequency}")
        
        
        
        print("Shape of Labeled Data:",x.shape)
        print("Shape of Unlabeled Data:",xu.shape)
        print("Accuracy on unlabelled images:",accuracy_score(labels,yu)*100)


        confidence = np.array([np.max(i) for i in probs])
        confidence_list.append(confidence)
        print(i,probs.shape)

        # Getting indices of probabilities in decreasing order
        idx = np.argsort(confidence)
        idx = idx[::-1] 

        # plt.yscale("log")
        # plt.plot(confidence[idx])
        
        
        print("Num img per class =",img_per_class)

        pop_list = [] #list of indices of images to be added
        label_count = []

        for j in idx:
            if confidence[j]>threshold and label_count.count(labels[j])<img_per_class:
                pop_list.append(j)
                label_count.append(labels[j])
        
        print("Number of images getting transferred:", len(pop_list))
        print('Accuracy of Pseudo-labelled img added to dataset:', accuracy_score(labels[pop_list],yu[pop_list])*100)
        pl_accuracies.append(accuracy_score(labels[pop_list],yu[pop_list])*100)

        # Confusion Matrix & Classwise Accuracies
        cmidx = [[(map(yu[j],i), map(labels[j],i)) for j in range(len(labels)) if labels[j]==i] for i in range(len(classes))]
        
        
        fig, ax = plt.subplots(math.ceil(len(classes)/3),3, figsize=(20, 5*math.ceil(len(classes)/3)))
        pltnum = 0
        cacc = []
        for cmid in cmidx:
            cacc.append(accuracy_score([z[0] for z in cmid], [z[1] for z in cmid])*100)
            if any(1 in x for x in cmid):
                ax[pltnum//3,pltnum%3].set_title(f"Class {pltnum}")
                confusion_matrix = metrics.confusion_matrix([z[0] for z in cmid], [z[1] for z in cmid])
                cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
                try:
                    cm_display.plot(ax=ax[pltnum//3,pltnum%3])
                except: 
                    pass
            pltnum += 1
        classwise_accuracies.append(cacc)
        fig.suptitle(f"Iteration {i}")
        plt.savefig(save_path+f'CM_{i}.png')
            
        

        if len(pop_list)<50:
            break
        
        x = np.append(x,xu[pop_list], axis=0)
        y = np.append(y,labels[pop_list], axis=0)
        yOracle = np.append(yOracle,yu[pop_list], axis=0)
        xu = np.delete(xu,pop_list, axis=0)
        yu = np.delete(yu,pop_list, axis=0)

        # Deleting variables
        del u_noisy_labels
        del cage

        val_scores.append(cnnScore(x, y,dataset,classes)[1]*100)
        oracle_val_scores.append(cnnScore(x,yOracle,dataset,classes)[1]*100)
        print(f"CNN Val accuracy trained on Lake Set for iteration {i}: ", val_scores[i])
        print(f"CNN Val accuracy trained on Oracle for iteration {i}: ", oracle_val_scores[i])
        # print(f"Classwise Precisions for iteration {i}:",classwise_accuracies[i])

        if yu.shape[0]<50:
            break

        
        print("="*135)


    return x,y,xu,yu,confidence_list, val_scores, oracle_val_scores, classwise_accuracies, pl_accuracies, baseline*len(val_scores)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classes', nargs="+", help="List of Classes")
    parser.add_argument('--models', nargs="+", help="List of Models")
    parser.add_argument('--random', type=int, default=1, help="Seed set is randomly generated. 1 or 0")
    parser.add_argument('--seed_set', type=str, default="C", help="Seed set(s) can be of three types Single (S), Individual (I), Compounding (C) (default=C)")
    parser.add_argument('--binary_model', type=int, default=1, help="LF Models are trained on only 2 classes. 1 or 0")
    parser.add_argument('--fraction', type=float, default=0.07, help="Fraction of the training data which is used as the seed set")
    parser.add_argument('--data_path', type=str, default="C:\\Users\\adity\\Documents\\GitHub\\MICCAI\\data\\bloodmnist.npz", help="Path to the Blood MNIST dataset")
    parser.add_argument('--save_path', type=str, default="C:\\Users\\adity\\Documents\\GitHub\\MICCAI\\data\\lfmodels2\\", help="Path to the directory where LF models, plots and spreadsheets will be saved")
    parser.add_argument('--max_iters', type=int, default=7, help="No. of iterations of Cage Loop")
    parser.add_argument('--threshold', type=float, default=10**-100, help="Threshold for Cage's confidence")
    parser.add_argument('--image_cap', type=int, default=200, help="Maximum no. of images per class taken out of lake set in a an iteration")


    args = parser.parse_args()

    # Exception Handling
    classes = []
    models = []
    try:
        classes = [int(c) for c in args.classes]
        models = [int(m) for m in args.models]
    except:
        raise Exception("Class and Model numbers can only be integers")
    
    for c in classes:
        if c<0 or c>=8:
            raise Exception("Class numbers should be between 0 and 7 only")
    
    for m in models:
        if m<0 or m>=5:
            raise Exception("Model numbers should be between 0 and 4 only")
    
    # print(classes)

    assert args.seed_set.upper() in ["C","I","S"], "Seed set can be only of type Compounding (C), Individual (I) or Single (S)"
    assert args.fraction >0 and args.fraction < 1, "Fraction of seed set can only be a number between 0 and 1"
    assert os.path.exists(args.data_path), "The Blood MNIST dataset does not exist at this path"
    random = bool(args.random)
    binary_model = bool(args.binary_model)

    LFS, ClassLabels = createLFs(classes,models,binary_model,args.save_path)

    if args.seed_set.upper()=="C" and random:
        x,y,xu,yu,confidence_list, val_scores, oracle_val_scores, classwise_accuracies, pl_accuracies, baseline = cage_loop_multiseed(LFS, args.max_iters, args.threshold, args.image_cap, classes,args.fraction,args.data_path,args.save_path,ClassLabels, True, binary_model)
    
    elif args.seed_set.upper()=="I" and random:
        x,y,xu,yu,confidence_list, val_scores, oracle_val_scores, classwise_accuracies, pl_accuracies, baseline = cage_loop_multiseed(LFS, args.max_iters, args.threshold, args.image_cap, classes,args.fraction,args.data_path,args.save_path,ClassLabels, False, binary_model)

    elif (args.seed_set.upper()=="C" or args.seed_set.upper()=="I") and not random:
        raise Exception("Multiple seed sets are always random, hence non-random seed set cannot be picked with seed set of type 'C' or 'I'")
    
    elif args.seed_set.upper()=="S":
        x,y,xu,yu,confidence_list, val_scores, oracle_val_scores, classwise_accuracies, pl_accuracies, baseline = cage_loop(LFS, args.max_iters, args.threshold, args.image_cap, classes,args.fraction,args.data_path,args.save_path,ClassLabels, random, binary_model)

    df = pd.DataFrame([baseline,val_scores,oracle_val_scores,pl_accuracies], columns=[f'Iteration {i}' for i in range(len(val_scores))], index=['Baseline','Our Results','Oracle/Skyline','Pseudo Labelled'])

    df2 = pd.DataFrame(classwise_accuracies, columns=[f'Class {i}' for i in range(len(classes))], index=[f'Iteration {i}' for i in range(len(val_scores))])

    df.to_csv(args.save_path+"results.csv")
    df2.to_csv(args.save_path+"classwise.csv")