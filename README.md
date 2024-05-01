##  SMS Spam Detection Machine Learining Detector
This project implements a machine learning model to classify SMS messages as spam or legitimate (ham).

### Abstract
The goal of this research was to use machine learning techniques to create an effective SMS spam detection system. An extensive preprocessing approach that included cleaning, transformation, and feature engineering was used to create a dataset of SMS messages that had been classified as authentic or spam. The selection of the Random Forest (RF) and XGBoost (XGB) algorithms for classification tasks was based on their effectiveness in managing high-dimensional data. To evaluate the performance of the model, evaluation criteria such as accuracy, precision, recall, and F1 score were used. More encouraging outcomes were shown by Random Forest in correctly identifying spam communications from authentic ones. By demonstrating how well machine learning algorithms analyze SMS data for classification, this effort advances the field of spam detection.

### Background:
Due to the widespread use of mobile phones, Short Message Service (SMS) has become the main form of communication on a global scale. Unsolicited messages, or spam, present a problem for SMS users because they might include unwanted or fraudulent material and put them at risk for identity theft and financial schemes.
Problem Statement:
For mobile consumers, the proliferation of SMS spam is a serious annoyance and security risk. It is crucial to distinguish and eliminate spam communications from authentic ones in order to improve user experience, safeguard privacy, and reduce security threats. Because spam messages are dynamic, traditional rule-based approaches frequently fail to capture their essence and require more sophisticated, adaptive strategies.

###  Getting Started
This project requires the following Python libraries:

* pandas (pd)
* matplotlib.pyplot (plt)
* seaborn (sns)
* pickle (for saving/loading the model - optional)
* nltk

**Installation:**

```bash
pip install pandas matplotlib seaborn nltk
```

###  Usage

1. Clone this repository:

```bash
git clone https://github.com/your_username/sms_spam_detection.git
```

2. Navigate to the project directory:

```bash
cd sms_spam_detection
```

3. Run the script:

```bash
python main.ipynb
```

This will perform data loading, cleaning, exploration, feature engineering, and visualization.

**Note:** The script currently includes model training and saving functionality.

###  Data

The dataset used in this research was sourced from the well-known data science competition and dataset website Kaggle.com.
SMS Spam Collection Dataset (kaggle.com)

###  Project Structure

```
sms_spam_detection/
├── Models/
│   └── rf_model.sav  # Previously Random Forest trained model
│   └── xgb_model.sav  # Previously XGBoost trained model
├── Results/
│   └── test_predictions.csv  # .csv with the predicted values for testing
├── requirements.txt  # List of required Python libraries
└── main.ipynb  # Notebook with the code
```
### Data Preprocessing Steps:
1. Removing Duplicates:
To protect the integrity of the data and avoid bias during analysis, duplicate entries in the dataset were found and eliminated. Overfitting can occur when machine learning models are trained with duplicate SMS messages.

2. Dimensionality Reduction:
Since the dataset mostly consists of textual data, dimensionality reduction techniques like Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) were not appropriate in this situation. However two unnamed columns were identified which had over 90% null data and were dropped.

3. Renaming Columns:
To make data manipulation easier and to increase readability, column names were standardized. Adding descriptive labels to columns improves the dataset's readability and makes further analysis easier.

4. Under sampling the ham data
Under sampling techniques were especially applied to the non-spam (ham) portion of the sample in order to rectify the imbalance caused by the higher proportion of spam messages compared to legal ones. The dataset's equilibrium was restored by arbitrarily choosing a subset of ham communications that was the same size as the spam messages. By ensuring that the machine learning models could learn from both classes without being biased towards the majority class (spam), our technique improved the overall performance and dependability of the SMS spam detection system.

5. Removing Punctuation:
The text messages were edited to remove punctuation, including exclamation points, periods, and commas. Eliminating punctuation makes the text easier to read and makes feature extraction less noisy.

### Future Engineering
Following data cleaning, we proceed to feature engineering. The initial step entails encoding the 'spam' feature using scikit-learn's LabelEncoder class.

6. Label Encoding
To make the categorical categories compatible with machine learning algorithms, they were converted into numerical values that indicate whether a communication is spam or real. Model training and evaluation are made easier by label encoding, which gives each category a distinct numerical identification. To enhance the array of features utilized by the model, we can incorporate two supplementary numerical attributes: word count and message length.

Message Length: This attribute indicates the quantity of characters present in the message.
Words: This attribute signifies the count of words within the message.

7. Tokenization:
The process of tokenizing text involves dividing it into smaller pieces known as tokens, which are usually words or subwords. Every token in the text denotes a unique unit of meaning. An essential first step in natural language processing (NLP) activities is tokenization, which makes it possible to analyze and comprehend textual input.

8. Removing Stop Words:
Stop words are frequently used terms that have minimal semantic significance and can be safely disregarded while doing an analysis. Stop words contain things like "the," "and," "is," and so on. Eliminating stop words increases the effectiveness of text processing algorithms and lowers noise in the dataset.

9. Stemming/Lemmatization using NLTK:
Lemmatization and stemming are methods for breaking words down to their most basic or root forms. Lemmatization is the process of mapping words to their dictionary form (lemma), whereas stemming is eliminating suffixes from words to retrieve their stems. For lemmatization and stemming, the Python Natural Language Toolkit (NLTK) package was used, guaranteeing uniformity and standardization of word forms throughout the dataset.

10. Using Trigrams and Bigrams to Gain Contextual Understanding:
We capture not just single words but also word sequences of two or three consecutive words by using NLTK's ngrams function to extract bigrams and trigrams from the preprocessed text. By taking into account the context in which words appear, this method helps the model better understand the underlying semantics and increase classification accuracy.

11. Using CountVectorizer to Represent Bigrams and Trigrams:
CountVectorizer was used to transform the extracted bigrams and trigrams into numerical representations appropriate for machine learning methods. In this method, a sparse matrix was created, with columns signifying distinct bigrams or trigrams and their corresponding counts, and rows representing individual messages. These attributes provide the model access to higher-order linguistic patterns, which improves its capacity to distinguish between spam and authentic messages.


### Modeling

Rationale for the Algorithms: 
XGBoost and Random Forest algorithms were chosen because to their propensity to handle textual data, which is a feature of the dataset. These algorithms are appropriate for our project because they have proven to perform well in text categorization tasks.

Parameter Technique used: GridSearchCV
GridSearchCV is a machine learning hyperparameter tuning technique. Using cross-validation, it methodically looks through a predetermined grid of hyperparameters to find the ideal combination. It selects the model with the highest performance after training and assessing it for every possible combination of hyperparameters. Lastly, it refines the model for deployment using the best hyperparameters across the board.
Here, it was used to adjust the Random Forest model's parameters in order to maximize its performance. The number of cross-validation folds (cv) and the intended evaluation metric (scoring) were defined for the parameter grid (param_grid). Since the F1 score offers a fair comparison of precision and recall, the refit parameter was set to 'f1' in order to choose the model based on the score.

### Challenges
Multiple challenges were encountered during the preprocessing, the major challenge was with stemming and lemmatization as we were using these steps, these functions were converting some of the words to Latin root which then were creating problem during analysis as we are analysing in English. The problem within coding were identified after rigorous inspection.
Not only this but also the accuracy parameters were not apt as the confusion matric was showing lesser true positive as the Precision and recall were giving 99% results. The problem was with the model, the machine were merging somehow the values of both models. It was later scrutinized.


### Results
By evaluating the trained models and concentrating on important performance measures including accuracy, precision, recall, and F1 score, experimental findings were achieved. Regarding the Random Forest (RF) paradigm:

* Precision: 90.46%
* Accuracy: 93.33%
* Recall: 86.82%
* F1 Rating: 89.96%

These show that the RF model was highly accurate in differentiating between SMS messages that were spam and those that were authentic. The model successfully reduced false positives with a precision of 93.33%, and its recall of 86.82% indicates that it can recognize most spam messages correctly. The balanced F1 score of 89.96% highlights how well the RF model is overall at identifying SMS spam.

The RF model outperformed the XGB model in terms of precision and overall accuracy, while the XGB model obtained complete recall, meaning it correctly detected every spam message. A larger rate of false positives is suggested by the poorer precision of 55.84%, which could cause valid messages to be mistakenly classified as spam. 

Overall, the findings demonstrate how well the Random Forest model performs in correctly identifying SMS messages as authentic or spam. The durability and appropriateness of the RF model for practical implementation in SMS spam detection systems is demonstrated by its balanced performance across many parameters.

### Conclusion
In conclusion, our study successfully developed an SMS spam detection system using Random Forest machine learning techniques. Utilizing state-of-the-art techniques such as NLTK's bigrams and trigrams, the model demonstrated great performance in accurately categorizing spam messages while minimizing false positives.

The results demonstrate how well machine learning methods work to address the persistent issue of SMS spam. The Random Forest model's remarkable recall, accuracy, and precision highlighted its suitability for real-world application in SMS communication networks. Furthermore, the model showed balanced performance with low false positives (FP) and high true positives (TP), recognizing 125 real messages with only 17 false positives. In addition, the model achieved high true negatives (TN) and low false negatives (FN) by correctly classifying 112 spam messages and avoiding 8 false negatives.

Considering the computational complexity and time commitment associated with model training, it makes sense to save the learned model for future use. Consequently, the model will be serialized using the Pickle library to facilitate efficient deployment and application in real-world scenarios.

The framework created by this project can be expanded upon by later research and development projects. It is possible for future generations of SMS spam detection systems to further enhance user security, privacy, and the overall quality of mobile communication by exploring new features, refining model designs, and resolving moral dilemmas.

###  Contributing

We welcome contributions to this project! Please create a pull request with any improvements or additions.

###  License

This project is licensed under the MIT License. See the LICENSE file for details.
