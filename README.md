##  SMS Spam Detection 

This project implements a machine learning model to classify SMS messages as spam or legitimate (ham).

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
python sms_spam_classifier.py
```

This will perform data loading, cleaning, exploration, feature engineering, and visualization.

**Note:** The script currently does not include model training or saving functionality.

###  Data

The project uses the `spam.csv` dataset containing labeled SMS messages.

###  Project Structure

```
sms_spam_detection/
├── data/
│   └── spam.csv  # SMS message dataset
├── src/
│   └── sms_spam_classifier.py  # Script for data processing and visualization
├── requirements.txt  # List of required Python libraries
└── README.txt  # This file (project documentation)
```

###  Contributing

We welcome contributions to this project! Please create a pull request with any improvements or additions.

###  License

This project is licensed under the MIT License. See the LICENSE file for details.
