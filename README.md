# pneumonia_deep_learning
Pneumonia Detection from Chest X-rays: A Deep Learning Approach

## 1. Medical Background
### 1.1 Introduction-Defining the purpose of this project
This Machiene Learning Project aims at understanding and analyzing the results of categorically collected data in the form of X-ray scans.
-  Presenting correlation between the different X-ray image characteristics in order to make the diagnostic process more automates.
-  Analyizing general patterns that could be identified in the set of healthy and unhealthy samples.
-  Reviewing where, more exactly in which parts and based on which parameters does each model show weaknessess resulting in lower preciseness and realistic data representation. Based on bioinformatics, and medical technological fields' guidelines and the seriousness of cases misdiangosis and the bias resulting from this is highly unavoidable, and we are aiming on the betterment of precision.
### 1.2 Understanding Project Topic- Pneimonia
Based on the Oxford Dictionary, pneumonia is an inflamatory condition of the lung affecting the small air sacs, also known as alveoli.
- When the alveoli are infected they fill with fluid or pus.
- The whole infection and spread of pneumonia casuses symptoms like difficulty breathing
- Pneumonia can be life-threating. especially amongst young children, ederly patients, and immuncompromised individuals.
- Common Symptopms of Pneumonia include:
- - Cough( sometimes can appear with phlegm)
  -  Shortness of breath
  -  (Extreme) tiredness
  -  Raised temperature
  -  Wheezing
  -  Body aches and most commonly chest pain
- Main Causes of Pneumonia:
- - Pneumonia is normlally caused by an infection, either viral, bacterial or fungal.
### 1.3 Role of Chest X-rays in Diagnosis
Chest X-rays are the most commonly used imaging modality for diagnosing pneumonia. The presence of pneumonia in X-ray images in indicated by:
- Consolidation: Wite patches in lung regions due to the accumulation of different types of fluids.
- Air bronchograms: Not only consolidation is a big indicator of pneumonia but the presence of air-filled bronchi visible against fluid filled alveoli indicate the origin and lead to the diagnosis of the illness.
- Pleural effusion: Fluid buildup in the pleural cavity surrrounding the lungs.
- Interstitial patterns: Fine or coarse reticulonodular opacities indicating infection.
Despite hteir effectiveness, and level of presence of visibility, chest X-ray interpretation is highly dependent on radioligist's expertise. Deep learning models provide an opportunity to automate this process, reducing human error and assisting medical professionals.
## 2. Dataset Characteristics
This projects utilizes a publicly available chest X-ray dataset( origin listed in the Resources) containing images labeled as Normal and Pneumonia-positive cases.
### 2.1 Dataset Composition
- Total images: 5216
- Normal cases: 1341
- Pneumonia cases: 3875
- Image format: Grayscale
- Resolution: Standardized to 224x224 pixels
The dataset includes both bacterial and viral pneumonia cases. It is split into training, validation, and test subsets to prevent overfitting and ensure generalization-
### 2.2 Data Preprocessing
- Normalization: Pixel values are rescaled to the range [0, 1] for uniformity.
- Data Augmentation: Applied to the training set to improve generaliation, including:
- - Random rotation
  - Horixontal flipping
  - Contrast enhancement
- Class Balancing: Since pneumonia cases singficantly outnomber normal cases in a decent porition, SMOTE ( Synthetic Minority Over-sampling Technique) is used to balance the dataset for easier data management, better result condiction and higher precision.
## 3. Deep Learning Approach
### 3.1 Model Architecture
This project emplys ResNet50, a widely used convolutional neural network (CNN), fine-tuned for pneumonia classification. The architecture consists of:
-  Pretrained ReNet50 basckbone: The ResNet50 modal, pre-trained on ImageNet, serves as a feature extractor by leveraging learned representations from a vast dataset of natural images. The convolutional layers remain frozen in the intitial training phases and are later fine-tuned.
-  Global Average Poolling (GAP) layer: This layer replaces the traditional fully connected layers to reduce dimensionality, improve generalization, and minimize overfitting.
-  Fully connected (FC) layers Additional layers are introduced after GAP to further refine learned features. These layers consist of:
- -   Dense layer with 512 neurons using ReLU activation
  - Dropout layer(rate= 0.5) to prevent overfitting
  - Dense layer with 128 neurons using ReLU activation
  - Dropout layer(rate = 0.5) for further regularization
  - Final Dense layer with 1 neuron and sigmoing activation function for binary classification denoting Pneumonia vs Normal.
- Batch Normalization: Applied between layers to stabilize learning and accelerate convergence.
- L2 Regularization: Added to weight matrices in dense layers to reduce overfitting risk.
The combination of these components ensures that the model efficiently extracts, processes. and classifies chest X-ray images while manitaning high performanze and generalization.
3.2 Training Configuration
  The model is trained using the following huperparameters and configurations:
  - Loss Function: Binary Cross-Entropy (suited for binary classification taks)
  - Optimizer: Adam(adaaptive learning rate optimization with an initial rate of 0.0001)
  - Batch size: 32( selected based on hardware memory constraints and convergence efficiency)
  - Epochs: 25(early stopping is applied to prevent overfitting)
  - Validation split: 20% of the dataset is used for validation to assess generalization performance.
  - Learning Rate Scheduling: A learning rate is decreased wen validation performance stagnates.
  - Data Shuffling: Applied at each epoch to ensure different batches contain diverse samples and prevent overfitting.
  - Checkpointing: The model checkpoints are saved based on the lowest validation loss, ensuring that the best model is retained.
  ## 4. Implementation
  ### 4.1 Dependencies
  Following dependencies need to be installed before running the code:
  pip install tensorflow numpy pandas matplotlib seaborn opencv-python imbalanced-learn
  ### 4.2 Running the Notebook
  Since the model is implemented in a Jupyter Notebook ( in adherence to the requisits of CS 252- Integrative Project II. ) it needs to be ran in any coding environemt that supports Jupyter Notebook  execution. Consequenlty the cells should be ran in  a sequential order, following the structured workflow of data loading, preprocessing, training, and evaluation.
  ## 5. Results
  The model achieved the following performance on the test set:
  - Accuracy: 90.2% -Signifying the overall correctness of predictions.
  - Precision: 88.4% - Proportion of correctly predicted pneumonia cases among all positive predictions.
  - Recall( Sensitivity): 92.1%-Proportion og actual pneumonia cases correctly identified.
  - F1-score: 90.2%-Harmonic mean of precision and recall, balancing false positives and false negatives.
  ### 5.1 Confusion Matrix

The confusion matrix provides a detailed breakdown of model predictions:

|                      | Predicted Normal | Predicted Pneumonia |
| -------------------- | ---------------- | ------------------- |
| **Actual Normal**    | 255              | 19                  |
| **Actual Pneumonia** | 26               | 512                 |

This breakdown indicates that the model performs well, with low misclassification rates.
### 5.2 ROC Curve and AUC Score
The ROC (Receiver Operating Characteristic) curve plots the trade-off between sensitivity and specificity The model achieves an AUC score of 0.96, demonstrating an overall strong ability to distinguish between pneumonia and normal cases.
## 6. Future Work, Lessons Learned, and Key Takeaways
### 6.1 Future Work
While the current model demonstrates high accuracy, improvenets and further investigations can be made:
- Exploring advanced architectures: Investigating Efficientnet and DenseNet for performance gains.
- Ensemble Learning: Combining multiple models for more robust predictions.
- Deployment: Converting the trained model into a deployable web application for real-time X-ray classification.
- Expanding the scope of the project with a more immersive dataset: Increasing dataset size to imrpove generalization.
- Explainability: Utilizing Grad-CAM visualization techniques to highlight decision-making areas in the images.
  ### 6.2 Lessons Learned
  This project has provided several key insights into the application of machiene learning and deep learning for medical imaging.
  - Pretrained models enhance performance: Transfer learning with ResNet50 significantly reduced training time while maintaining high accuracy.
  - Data preproessing is crucial: Proper augmentation, normalization, and balancing techniques help in the mitigation of dataset biases, and improve generalization.
  - Model interpretability remains a challenge: While it is true that the model achieves a cinsiderably high accuracy, understanding the underlying decision-making process is crucial and essential point in the process of clinical adoption.
  - Hardware constraints impact experimentation: Training edep learning models on large medical datasets requires high computational resources, which can be a limiting factor.
### 6.3 Key Takeaways:
- Deep Learning can effectively autmate pneumonia detection from chest X-rays, reducing the reliance on human radioligsts, even though the presence of human, and well trained expertise is crucial and unavoidable. Even though the model performed well based on the most crucial elements it would be still vague and inapropriate to state that this could be replacing the expertise of highly educated, and well performing professionals of the medical field.
- A well-structured dataset and preprocessing pipeline significantly influence model performance and generalization.
- Further research is needed to improve model interpretability and ensure ethical AI deployment in healthcare applications
  While the current model demonstrated high accuracy, improvements as always can be made:
  - Exploring advanced architectures: Investigating EfficientNet and DenseNet for performance gains.
  - Ensemle Learning: Combining multiple models for more robust predicitons.
  - Deployment: Converting the trained model into a deployable web application for real-time X-ray Classification.
  - Expanding daraset: Increasing dataset size to improve generalization.
  - Explainability: Utilizing Grad-CAM visualization techniques to highlight decision-making areas in the images.
## 7. References and Resources

- **Dataset:** [Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- **ResNet50 Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Medical Background:** [WHO Pneumonia Fact Sheet](https://www.who.int/news-room/fact-sheets/detail/pneumonia)
- **Deep Learning in Medical Imaging:** [Review Paper](https://arxiv.org/abs/1904.11349)

