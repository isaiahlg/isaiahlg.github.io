################################
##
## Simple Examples of 
## ANNs, RNNs (and LSTM, BRNN), and CNNs
## in Python/Keras
##
##########################################
## Gates
# https://gatesboltonanalytics.com/?page_id=903

from tensorflow.keras.layers import Activation
 #https://www.tensorflow.org/api_docs/python/tf/keras/Input
import numpy as np
import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import tensorflow.keras
#from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras import layers



from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os

from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string

from bs4 import BeautifulSoup
from collections import Counter

from nltk.corpus import stopwords
#nltk.download('stopwords')
from nltk.tokenize import word_tokenize



## These settings show/print dfs in diff ways
pd.set_option('display.max_columns', 50)
pd.set_option('display.width', 50)
#pd.options.display.max_seq_items = None
#pd.pandas.set_option('display.max_columns', None)
pd.options.display.max_colwidth = 50


################
##
## ANN - CNN - RNN
## Movies Dataset
## https://www.kaggle.com/datasets/columbine/imdb-dataset-sentiment-analysis-in-csv-format
##
####  Gates
#################################################################################
## 
## Update the path for YOUR computer
##
##################################
path = "C:/Users/profa/Desktop/UCB/NNCSCI5922/Code/IMDB_Movie_Datasets_Train_Test_Valid/"

TrainData = pd.read_csv(str(path+"Train.csv"))
# print(TrainData.shape)
print(TrainData.head(10))
# print(type(TrainData))

TestData = pd.read_csv(str(path+"Test.csv"))
#print(TestData.shape)
TestData.head(10)

ValidData = pd.read_csv(str(path+"Valid.csv"))
#print(ValidData.shape)
ValidData.head(10)

## Concat requires a list
## Place all data from above into one dataframe
FullDataset=pd.concat([TrainData,TestData, ValidData])
#print(FullDataset.shape)
## Clean Up TrainData
## Get the vocab

#print(TrainData.head())
# Testing iterating the columns 
for col in TrainData.columns: 
    print(col) 
    
## Check Content   --------------------
#print(TrainData["text"])
#print(TrainData["label"]) ##0 is negative, 1 is positive

### Tokenize and Vectorize 
## Create the list 
## Keep the labels

ReviewsLIST=[]  ## from the text column
LabelLIST=[]    

for nextreview, nextlabel in zip(TrainData["text"], TrainData["label"]):
    ReviewsLIST.append(nextreview)
    LabelLIST.append(nextlabel)

print("A Look at some of the reviews list is:\n")
print(ReviewsLIST[0:20])

print("A Look at some of the labels list is:\n")
print(LabelLIST[0:20])


######################################
## Optional - for Stemming the data
##
################################################
## Instantiate it
A_STEMMER=PorterStemmer()
## test it
print(A_STEMMER.stem("fishers"))
#----------------------------------------
# Use NLTK's PorterStemmer in a function - DEFINE THE FUNCTION
#-------------------------------------------------------
def MY_STEMMER(str_input):
    ## Only use letters, no punct, no nums, make lowercase...
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [A_STEMMER.stem(word) for word in words] ## Use the Stemmer...
    return words

#########################################
##
##  Build the labeled dataframe
##  Get the Vocab  - here keeping top 10,000
##
######################################################

### Vectorize
## Instantiate your CV
MyCountV=CountVectorizer(
        input="content",  
        lowercase=True, 
        #stop_words = "english", ## This is optional
        #tokenizer=MY_STEMMER, ## Stemming is optional
        max_features=11000  ## This can be updated
        )

## Use your CV 
MyDTM = MyCountV.fit_transform(ReviewsLIST)  # create a sparse matrix
#print(type(MyDTM))


ColumnNames=MyCountV.get_feature_names() ## This is the vocab
#print(ColumnNames)
#print(type(ColumnNames))

## Here we can clean up the columns


## Build the data frame
MyDTM_DF=pd.DataFrame(MyDTM.toarray(),columns=ColumnNames)

## Convert the labels from list to df
Labels_DF = pd.DataFrame(LabelLIST,columns=['LABEL'])

## Check your new DF and you new Labels df:
# print("Labels\n")
print(Labels_DF)
# print("DF\n")
print(MyDTM_DF.iloc[:,0:20])
print(MyDTM_DF.shape) ## 40,000 by 11000

############################################
##
##  Remove any columns that contain numbers
##  Remove columns with words not the size 
##  you want. For example, words<3 chars
##
##############################################
##------------------------------------------------------
### DEFINE A FUNCTION that returns True if numbers
##  are in a string 
def Logical_Numbers_Present(anyString):
    return any(char.isdigit() for char in anyString)
##----------------------------------------------------

for nextcol in MyDTM_DF.columns:
    #print(nextcol)
    ## Remove unwanted columns
    #Result=str.isdigit(nextcol) ## Fast way to check numbers
    #print(Result)
    
    ##-------------call the function -------
    LogResult=Logical_Numbers_Present(nextcol)
    #print(LogResult)
    ## The above returns a logical of True or False
    
    ## The following will remove all columns that contains numbers
    if(LogResult==True):
        #print(LogResult)
        #print(nextcol)
        MyDTM_DF=MyDTM_DF.drop([nextcol], axis=1)

    ## The following will remove any column with name
    ## of 3 or smaller - like "it" or "of" or "pre".
    ##print(len(nextcol))  ## check it first
    ## NOTE: You can also use this code to CONTROL
    ## the words in the columns. For example - you can
    ## have only words between lengths 5 and 9. 
    ## In this case, we remove columns with words <= 3.
    elif(len(str(nextcol))<3):
        print(nextcol)
        MyDTM_DF=MyDTM_DF.drop([nextcol], axis=1)
       
    

##Save original DF - without the lables
My_Orig_DF=MyDTM_DF
print(My_Orig_DF.head(10))



## Now - let's create a complete and labeled
## dataframe:
dfs = [Labels_DF, MyDTM_DF]
print(dfs)
print("shape of labels\n", Labels_DF)
print("shape of data\n", MyDTM_DF)

Final_DF_Labeled = pd.concat(dfs,axis=1, join='inner')
## DF with labels
print(Final_DF_Labeled.iloc[:, 0:2])
print(Final_DF_Labeled.shape)


################################################
## FYI
## An alternative option for most frequent 10,000 words 
## Not needed here as we used CountVectorizer with option
## max_features
# print (df.shape[0])
# print (df[:10000].value.sum()/df.value.sum())
# top_words = list(df[:10000].key.values)
# print(top_words)
# ## Example using index
# index = top_words.index("humiliating")
# print(index)
##############################################

## Create list of all words
print(Final_DF_Labeled.columns[0])
NumCols=Final_DF_Labeled.shape[1]
print(NumCols)
print(len(list(Final_DF_Labeled.columns)))

top_words=list(Final_DF_Labeled.columns[1:NumCols+1])
## Exclude the Label

print(top_words[0])
print(top_words[-1])


print(type(top_words))
print(top_words.index("aamir")) ## index 0 in top_words
print(top_words.index("zucco")) #index NumCols - 2 in top_words

## Encoding the data
def Encode(review):
    words = review.split()
   # print(words)
    if len(words) > 500:
        words = words[:500]
        #print(words)
    encoding = []
    for word in words:
        try:
            index = top_words.index(word)
        except:
            index = (NumCols - 1)
        encoding.append(index)
    while len(encoding) < 500:
        encoding.append(NumCols)
    return encoding
##-------------------------------------------------------
## Test the code to assure that it is
## doing what you think it should 

result1 = Encode("aaron aamir abbey abbott abilities zucco ")
print(result1)
result2 = Encode("york young younger youngest youngsters youth youthful youtube zach zane zany zealand zellweger")
print(result2)
print(len(result2)) ## Will be 500 because we set it that way above
##-----------------------------------------------------------
 
###################################
## Now we are ready to encode all of our
## reviews - which are called "text" in
## our dataset. 

# Using vocab from above i -  convert reviews (text) into numerical form 
# Replacing each word with its corresponding integer index value from the 
# vocabulary. Words not in the vocab will
# be assigned  as the max length of the vocab + 1 
## ########################################################

# Encode our training and testing datasets
# with same vocab. 

print(TestData.head(10))
print(TestData.shape)
print(TrainData.shape)


############### Final Training and Testing data and labels-----------------
training_data = np.array([Encode(review) for review in TrainData["text"]])
print(training_data[20])
print(training_data.shape)

testing_data = np.array([Encode(review) for review in TestData['text']])
print(testing_data[20])

validation_data = np.array([Encode(review) for review in ValidData['text']])

print (training_data.shape, testing_data.shape)

## Prepare the labels if they are not already 0 and 1. In our case they are
## so these lines are commented out and just FYI
#train_labels = [1 if label=='positive' else 0 for sentiment in TrainData['label']]
#test_labels = [1 if label=='positive' else 0 for sentiment in TestData['label']]
train_labels = np.array([TrainData['label']])
train_labels=train_labels.T
print(train_labels.shape)
test_labels = np.array([TestData['label']])
test_labels=test_labels.T
print(test_labels.shape)

###############################
## ANN
#################################
## Simple Dense NN for sentiment analysis (classification 0 neg, 1 pos)
# First layer: Embedding Layer (Keras Embedding Layer) that will learn embeddings 
# for different words .
## RE: ## https://keras.io/api/layers/core_layers/embedding/
## input_dim: Integer. Size of the vocabulary
## input_length: Length of input sequences, when it is constant.
print(NumCols)   
input_dim = NumCols + 1 

 
My_ANN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=input_dim, output_dim=32, input_length=500),
  tf.keras.layers.Dense(32, activation='relu'), 
  tf.keras.layers.Dense(16, activation='relu'),
  tf.keras.layers.Dropout(.5), 
  tf.keras.layers.Dense(8, activation='relu'),
  tf.keras.layers.Dropout(.5), 
  tf.keras.layers.Dense(1, activation='sigmoid')
  
])
    
My_ANN_Model.summary()

loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
My_ANN_Model.compile(
                 loss=loss_function,
                 metrics=["accuracy"],
                 optimizer='adam'
                 )



print(training_data[0:3, 0:3])
print(training_data.shape)
print(train_labels[10])

Hist=My_ANN_Model.fit(training_data, train_labels, epochs=5, validation_data=(testing_data, test_labels))

##batch_size=256,

###############################################
##
## SimpleRNN
##
######################################################
#print(input_dim)
output_dim=32
print(input_dim)
print(training_data.shape)

My_SimpleRNN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim, input_length=500),
  tf.keras.layers.SimpleRNN(units =50, input_shape=(32,32,3)),
  ## If not using Embedding, you would use SimpleRNN(units, input_shape=(x,y))
  tf.keras.layers.Dense(1, activation='sigmoid')
])
    
My_SimpleRNN_Model.summary()

loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
My_SimpleRNN_Model.compile(
                 loss=loss_function,
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

print(training_data[0:3, 0:3])

print(train_labels[10])

Hist=My_SimpleRNN_Model.fit(training_data, train_labels, epochs=5, validation_data=(testing_data, test_labels))

###################################
## RNN with Bidirectional 
###############################################
# input_dim = NumCols + 1

 #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Bidirectional
My_RNN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=input_dim, output_dim=32, input_length=500),
  tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(50)),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
    
My_RNN_Model.summary()

loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
My_RNN_Model.compile(
                 loss=loss_function,
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

print(training_data[0:3, 0:3])
print(training_data.shape)
print(train_labels[10])

Hist=My_RNN_Model.fit(training_data, train_labels, epochs=5, validation_data=(testing_data, test_labels))


############################################
## LSTM
#############################################
My_LSTM_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=input_dim, output_dim=32, input_length=500),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50)),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
    
My_LSTM_Model.summary()

loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
My_LSTM_Model.compile(
                 loss=loss_function,
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

print(training_data[0:3, 0:3])
print(training_data.shape)
print(train_labels[10])

Hist=My_LSTM_Model.fit(training_data, train_labels, batch_size=12, epochs=5, validation_data=(testing_data, test_labels))


######################################
## CNN
########################################


##
My_CNN_Model = tf.keras.models.Sequential([
  tf.keras.layers.Embedding(input_dim=input_dim, output_dim=32, input_length=500),
  
  tf.keras.layers.Conv1D(50, kernel_size=3, activation='relu'),
  tf.keras.layers.MaxPool1D(pool_size=2),
  
  tf.keras.layers.Conv1D(40, kernel_size=3, activation='relu'),
  tf.keras.layers.MaxPool1D(pool_size=2),
  
  tf.keras.layers.Conv1D(30, kernel_size=3, activation='relu'),
  tf.keras.layers.MaxPool1D(pool_size=2),
  
  tf.keras.layers.Conv1D(30, kernel_size=3, activation='relu'),
  tf.keras.layers.MaxPool1D(pool_size=2),
  
  tf.keras.layers.Flatten(),
 
  tf.keras.layers.Dense(20),
  tf.keras.layers.Dropout(0.5),
 
  tf.keras.layers.Dense(1, activation="sigmoid")

  
  tf.keras.layers.Dense(1, activation='sigmoid')
])
    
My_CNN_Model.summary()

loss_function = keras.losses.BinaryCrossentropy(from_logits=False)
My_CNN_Model.compile(
                 loss=loss_function,
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

print(training_data[0:3, 0:3])
print(training_data.shape)
print(train_labels[10])

Hist=My_CNN_Model.fit(training_data, train_labels, batch_size=12, epochs=5, validation_data=(testing_data, test_labels))


##


print("Evaluate model on test data")
results = model.evaluate(testing_data, test_labels, batch_size=256)
print("test loss, test acc:", results)

# Generate a prediction using model.predict() 
# and calculate it's shape:
print("Generate a prediction")
prediction = model.predict(testing_data)
print(prediction)
print("prediction shape:", prediction.shape)
print(type(prediction))
prediction[prediction > .5] = 1
prediction[prediction <= .5] = 0
print(prediction)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(prediction, test_labels))

###################################################
##
## One more example
##
## An ANN used on image data that is Flattened
##
#####################################################
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
 
# Load the CIFAR-10 dataset
(training_data2, train_labels2), (testing_data2, test_labels2) = cifar10.load_data()
 
# Scale the pixel values to between 0 and 1
training_data2 = training_data2 / 255.0
testing_data2 = testing_data2 / 255.0
 
# Convert the labels to one-hot encoding
train_labels2 = to_categorical(train_labels2)
test_labels2 = to_categorical(test_labels2)

ANN_Model_Images = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
  tf.keras.layers.Dense(200, activation='relu'), 
  tf.keras.layers.Dense(150, activation='relu'),
  tf.keras.layers.Dense(10, activation="softmax"),  
])

ANN_Model_Images.summary()

ANN_Model_Images.compile(
                 loss="categorical_crossentropy",
                 metrics=["accuracy"],
                 optimizer='adam'
                 )

Hist= ANN_Model_Images.fit(training_data2, train_labels2, epochs=50, validation_data=(testing_data2, test_labels2))