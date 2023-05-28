# MICCAI

### File Descriptions
* [Tiny Models](https://github.com/akshitt/MICCAI/blob/main/tiny_models.ipynb) - Code to train simple ML models on BloodMNIST (on 40 images)    
* [BloodMNIST_40](https://github.com/akshitt/MICCAI/blob/main/bloodmnist_40.npz) - Trimmed BloodMNIST dataset with 40 train (5 per class) & 400 test images (50 per class) 


### Plan 
* Trim BloodMNIST dataset
* Train 7 basic ML classifier models on limited data 
* Feed these models as Labelling Functions into SPEAR for aggregation
* Label/Test on the remaining dataset
* Train & Test a DL Model on same data  
* Compare accuracies

### Links
* [Edit Overleaf doc](https://www.overleaf.com/1561435369grrnqdqrpzmy)

### Instructions to run Cage notebooks

1. Create a new virtual environment, navigate to this directory and run the following command:

    ```
    pip install -r requirements.txt
    ```

2. Make the following changes in Spear library:
    * In file "spear/cage/core.py" change line 153 to:
        ```
        if path_test != None:
            assert np.all(np.logical_and(y_true_test >= 0, y_true_test < self.n_classes))
        ```
    * In file "spear/labeling/analysis/core.py" change line 335 or 336 just find similar line and replace to:
        ```
        confusion_matrix(Y, self.L[:, i], labels=labels)[1:, 1:] for i in range(m)
        ```

3. Configure "path" and "save_path" in code/config.json according to your system

4. Select the virtual environment as the kernel while running the notebook
