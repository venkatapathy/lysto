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
5. 
spear/jl/core.py line line: 100
		
        elif self.feature_based_model =='resnet':
		        self.feature_model = ResNet(self.n_features, self.n_hidden, self.n_classes).to(device = self.device)
                
spear/jl/models/models.py line: 47

import torch.nn as nn
import torchvision.models as models
import torch

class ResNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Reshape the input tensor to have dimensions [batch_size, 3, 30, 30]
        x = x.view(-1, 3, 30, 30)
        out = self.resnet(x)
        out = nn.functional.relu(out)
        return self.out(out)
                
