#!/usr/bin/env python
# coding: utf-8

# # FakeNewsDetection
# 

# In this notebook, we will be solving a use case of 'Fake News Detection' - Predict if a news published in an article is 'True' or 'Fake' for a sample data using ML Algorithms

# ## Library Imports

# In[16]:


#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import nltk #Import NLTK ---> Natural Language Toolkit
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


# ## Data Description

# 
# The sample dataset we use is fakedata.csv which is available in the current directory. This can be replaced with any dataset containing a text column with 'News' and a label column citing is the news is 'True' or 'Fake'
# 
# Attributes of fakedata.csv dataset:
# 
# text: text of the article
# 
# label: a label that marks the article as potentially unreliable
# 
# 1: fake 0: true

# ## Variables 
# 

# In[1]:


#Specify the input filename
INPUTFILE = r'fakenews.csv'
#Specify the ratio of test data for prediction
test_data_ratio = 0.2


# ## Data Loading & Visualization

# In[4]:


#Load the file for analysis. Change the path of the file if required
df = pd.read_csv(INPUTFILE)


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


#Plot the distribution of the label column in the data
cat_target=df.label
label_size = [cat_target.sum(),len(cat_target)-cat_target.sum()]
plt.pie(label_size,explode=[0.1,0.1],colors=['darkgrey','darkorange'],startangle=90,shadow=True,labels=['1','0'],autopct='%1.1f%%')


# ## Data Preprocessing

# Since the input data is 'text', it is necessary to preprocess it. The steps in preprocessing include Tokenization, Stop word removal, Stemming, Tf-Idf Vectorizer. During this preprocessing phase, the data is cleansed,treated and converted to vectors for the machine/algorithm to understand!

# In[8]:


# create a function to preprocess the data
def preprocess_data(data):
  
  # 1. Tokenization
    tk = RegexpTokenizer('\s+', gaps = True)
    text_data = [] # List for storing the tokenized data
    for values in data.text:
        tokenized_data = tk.tokenize(values) # Tokenize the news
        text_data.append(tokenized_data) # append the tokenized data

  # 2. Stopword Removal

  # Extract the stopwords
    sw = stopwords.words('english')
    clean_data = [] # List for storing the clean text
  # Remove the stopwords using stopwords
    for data in text_data:
        clean_text = [words.lower() for words in data if words.lower() not in sw]
        clean_data.append(clean_text) # Appned the clean_text in the clean_data list
  
  # 3. Stemming

  # Create a stemmer object
    ps = PorterStemmer()
    stemmed_data = [] # List for storing the stemmed data
    for data in clean_data:
        stemmed_text = [ps.stem(words) for words in data] # Stem the words
        stemmed_data.append(stemmed_text) # Append the stemmed text
  

  # 4. tfidf vectorizer --> Term Frequency Inverse Document Frequency  
  # Flatten the stemmed data

    updated_data = []
    for data in stemmed_data:
        updated_data.append(" ".join(data))

  # TFID Vector object
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(updated_data)

    return tfidf_matrix


# In[70]:


# preprocess the text data
preprocessed_data = preprocess_data(df.drop('label', axis=1))


# The output of this preprocessing step is a matrix of vectors representing the input data. We concatenate this with the target column for subsequent steps

# In[62]:


features_df = pd.DataFrame(preprocessed_data.toarray())
datadf = pd.concat([features_df,df['label']],axis=1)


# ## Model Development

# We will be using Scikit-learn ML library for training and predicting. There are many algorithms available in Scikit-learn. We will be experimenting with Logistic Regression & NaiveBayes algorithms.

# In[12]:


print("---Preprocessing data---")
print(df.shape)
df.drop_duplicates(inplace=True)
print(df.shape)
df.dropna(inplace=True)
print(df.shape)


# ### Splitting into Training and Testing

# In[13]:


# Split data into train & test in the ratio of 80:20
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(datadf, datadf.label, test_size=test_data_ratio, random_state = 42)


# ### Training the Model

# In[79]:


# Import LogisticRegression model from sklearn library
from sklearn.linear_model import LogisticRegressionCV

# Initialize the model 
lr_reg = LogisticRegressionCV(Cs=20, cv=3, random_state=42)

# fit the model
lr_reg.fit(X_train, y_train)


# In[82]:


# Import Naive Bayes model from sklearn library
from sklearn.naive_bayes import MultinomialNB


# Initialize the model
mnb = MultinomialNB(alpha=0.1)

# Fit the model
mnb.fit(X_train, y_train)
with open("fakenewsmodel.pkl","wb") as file:
        pickle.dump(mnb,file)


# ### Evaluation Metrics

# In[20]:


# Evaluation Metrics for Classification models
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Function for deriving a classification report & Confusion matrix for any algorithm
def compute_metrics(data, y_true, model_obj, model):

        # Make predictions
    y_pred = model_obj.predict(data)
    print(metrics.classification_report(y_true, y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=model_obj.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_obj.classes_)
    disp.plot()
    plt.show()


# ## Logistic Regression

# In[80]:


#  Compute the Evaluation Metrics for train set of Logistic Regression model
lr_metrics_train =  compute_metrics(X_train, y_train, lr_reg, 'LogisticRegression')
lr_metrics_train


# In[81]:


# Compute the Evaluation Metrics for test set for Logistic Regression model
lr_metrics =  compute_metrics(X_test, y_test, lr_reg, 'LogisticRegression')
lr_metrics


# ## Naive Bayes

# In[21]:


print("Naive Bayes model trained successfully!")
print("Classification Metrics for the train set is:" + "\n")
compute_metrics(X_train, y_train, mnb, 'Naive Bayes')


# ### Testing the Model

# In[24]:


#  Compute the Evaluation Metrics for test set of Naive Bayes model
mnb_metrics = compute_metrics(X_test, y_test, mnb, 'Naive Bayes')
mnb_metrics


# ## Prediction on new Unlabelled Data

# In[74]:


with open("fakenewsmodel.pkl", "rb") as file:
       model=pickle.load(file)
df_test =pd.read_csv('fakenewstest.csv')
    


# In[73]:


print("---Preprocessing data---")
print(df_test.shape)
df_test.drop_duplicates(inplace=True)
print(df_test.shape)
df_test.dropna(inplace=True)
print(df_test.shape)


# In[78]:


preprocessed_testdata =preprocess_data(df_test)

print("shape of transform in testing is: ", preprocessed_testdata.shape)
features_df = pd.DataFrame(preprocessed_testdata.toarray())
print(features_df.shape)


# In[ ]:


df_test["label"]=model.predict(features_df)
      
model.predict_proba(features_df)
model.classes_
probabs = model.predict_proba(features_df)

probs = list()
for prob in probabs:
    probs.append(round(max(prob[0],prob[1]), 2))
df_test["probability"] = probs
df_test.to_csv("test_pred.csv",index=False)

