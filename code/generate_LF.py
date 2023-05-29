
import json
import pathlib
import requests
from utils import custom_random_dataset, train_all_LF

def get_variables():
    
    f = open("/home/venkat/lysto/code/config.json")
    data = json.load(f)   
    classes = data["classes"]
    label_frac = data["label_frac"]
    path = data["path"]
    save_path = data["save_path"]
    return classes,label_frac,path,save_path

def get_variables2():
    
    f = open("config2.json")
    data = json.load(f)   
    classes = data["classes"]
    label_frac = data["label_frac"]
    path = data["path"]
    save_path = data["save_path"]
    return classes,label_frac,path,save_path

if __name__ == "__main__":
    
    classes,label_frac,path,save_path = get_variables()
    # print(classes)
    num_cls = len(classes)
    dataset,x,y = custom_random_dataset(classes=classes, path=path, fraction=label_frac)
    train_all_LF(x,y,num_cls,path=save_path+str(int(label_frac*100)),fraction=label_frac)
