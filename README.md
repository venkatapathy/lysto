# CIKM 2023

TEST DATASET: https://drive.google.com/drive/folders/1aKb5JbnW3kZF_hNJ7XMIX8-EW7Yuwas-?usp=sharing


### Instructions to run JL notebooks

1. Create a new virtual environment, navigate to this directory and run the following command:

    ```
    pip install -r requirements.txt
    ```

    ```
    pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ submodlib
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
