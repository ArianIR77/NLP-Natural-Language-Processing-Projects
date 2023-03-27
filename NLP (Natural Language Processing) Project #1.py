#!/usr/bin/env python
# coding: utf-8

# # NLP (Natural Language Processing)

# ## SMS Spam Collection Data Set
# 

# In[1]:


import nltk


# In[2]:


#nltk.download_shell()


# ## SMS Spam Collection Data Set
# 

# ## Get the Data

# We'll be using a dataset from the [UCI datasets](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)!

# The file we are using contains a collection of more than 5 thousand SMS phone messages.

# use rstrip() plus a list comprehension to get a list of all the lines of text messages:

# In[3]:


messages = [line.rstrip() for line in open('smsspamcollection/SMSSpamCollection')]


# In[4]:


len(messages)


# In[5]:


messages[0]


# In[6]:


for msg_num,msg in enumerate(messages[:10]):
    print(msg_num,msg)
    print('\n')


# Due to the spacing we can tell that this is a [TSV](http://en.wikipedia.org/wiki/Tab-separated_values) ("tab separated values") file, where the first column is a label saying whether the given message is a normal message (commonly known as "ham") or "spam". The second column is the message itself. (Note our numbers aren't part of the file, they are just from the **enumerate** call).
# 
# Using these labeled ham and spam examples, we'll **train a machine learning model to learn to discriminate between ham/spam automatically**. Then, with a trained model, we'll be able to **classify arbitrary unlabeled messages** as ham or spam.

# In[7]:


import pandas as pd


# In[8]:


messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t',
                           names=["label", "message"])
messages.head()


# ## Exploratory Data Analysis

# In[9]:


messages.describe()


# In[10]:


messages.groupby('label').describe()


# make a new column to detect how long the text messages are:

# In[11]:


messages['length'] = messages['message'].apply(len)


# In[12]:


messages.head()


# ### Data Visualization
# 

# In[13]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


sns.set_style('whitegrid')
sns.histplot(x='length', data=messages, bins=150)


# In[15]:


messages['length'].describe()


# 910 characters, let's use masking to find this message:

# In[16]:


messages[messages['length']>900]['message'].iloc[0]


#  let's focus back on the idea of trying to see if message length is a distinguishing feature between ham and spam:

# In[17]:


messages.hist(column='length', by='label', bins=50,figsize=(12,4))


# Through just basic EDA we've been able to discover a trend that spam messages tend to have more characters. 

# ## Text Pre-processing

# the bag-of-words approach, where each unique word in a text will be represented by one number.
# 
# In this section we'll convert the raw messages (sequence of characters) into vectors (sequences of numbers).
# 
# As a first step, let's write a function that will split a message into its individual words and return a list. We'll also remove very common words, ('the', 'a', etc..). To do this we will take advantage of the NLTK library. It's pretty much the standard library in Python for processing text and has a lot of useful features.

# Let's create a function that will process the string in the message column, then we can just use apply() in pandas do process all the text in the DataFrame.
# 
# First removing punctuation. We can just take advantage of Python's built-in string library to get a quick list of all the possible punctuation:

# In[18]:


import string

mess = 'Sample message! Notice: it has punctuation.'

# Check characters to see if they are in punctuation
nopunc = [char for char in mess if char not in string.punctuation]

# Join the characters again to form the string.
nopunc = ''.join(nopunc)


# Remove stopwords

# In[19]:


from nltk.corpus import stopwords
stopwords.words('english')[0:10] # Show some stop words


# In[20]:


# Now just remove any stopwords
clean_mess = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[21]:


clean_mess


# Now let's put both of these together in a function to apply it to our DataFrame later on:

# In[22]:


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# Now let's "tokenize" these messages. Tokenization is just the term used to describe the process of converting the normal text strings in to a list of tokens (words that we actually want).

# In[23]:


# Check to make sure its working
messages['message'].head(5).apply(text_process)


# ## Vectorization

# Currently, we have the messages as lists of tokens (also known as [lemmas](http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)) and now we need to convert each of those messages into a vector the SciKit Learn's algorithm models can work with.
# 
# Now we'll convert each message, represented as a list of tokens (lemmas) above, into a vector that machine learning models can understand.
# 
# We'll do that in three steps using the bag-of-words model:
# 
# 1. Count how many times does a word occur in each message (Known as term frequency)
# 
# 2. Weigh the counts, so that frequent tokens get lower weight (inverse document frequency)
# 
# 3. Normalize the vectors to unit length, to abstract from the original text length (L2 norm)

# Let's begin the first step:
# 
# Each vector will have as many dimensions as there are unique words in the SMS corpus. We will first use SciKit Learn's CountVectorizer. This model will convert a collection of text documents to a matrix of token counts.

# In[24]:


from sklearn.feature_extraction.text import CountVectorizer


# In[25]:


bow_transformer = CountVectorizer(analyzer=text_process).fit(messages['message'])


# In[26]:


# Print total number of vocab words
print(len(bow_transformer.vocabulary_))


# Let's take one text message and get its bag-of-words counts as a vector, putting to use our new bow_transformer:

# In[27]:


message4 = messages['message'][3]
print(message4)


# Now let's see its vector representation:

# In[28]:


bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)


# This means that there are seven unique words in message number 4 (after removing common stop words). Two of them appear twice, the rest only once. Let's go ahead and check and confirm which ones appear twice:

# In[29]:


print(bow_transformer.get_feature_names_out()[4068])
print(bow_transformer.get_feature_names_out()[9554])


# Now we can use .transform on our Bag-of-Words (bow) transformed object and transform the entire DataFrame of messages. Let's go ahead and check out how the bag-of-words counts for the entire SMS corpus is a large, sparse matrix:

# In[30]:


messages_bow = bow_transformer.transform(messages['message'])


# In[31]:


print('Shape of Sparse Matrix: ', messages_bow.shape)
print('Amount of Non-Zero occurences: ', messages_bow.nnz)


# In[32]:


sparsity = (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1]))
print('sparsity: {}'.format((sparsity)))


# After the counting, the term weighting and normalization can be done with [TF-IDF](http://en.wikipedia.org/wiki/Tf%E2%80%93idf), using scikit-learn's `TfidfTransformer`.
# 
# ____
# ### So what is TF-IDF?
# TF-IDF stands for *term frequency-inverse document frequency*, and the tf-idf weight is a weight often used in information retrieval and text mining. This weight is a statistical measure used to evaluate how important a word is to a document in a collection or corpus. The importance increases proportionally to the number of times a word appears in the document but is offset by the frequency of the word in the corpus. Variations of the tf-idf weighting scheme are often used by search engines as a central tool in scoring and ranking a document's relevance given a user query.
# 
# One of the simplest ranking functions is computed by summing the tf-idf for each query term; many more sophisticated ranking functions are variants of this simple model.
# 
# Typically, the tf-idf weight is composed by two terms: the first computes the normalized Term Frequency (TF), aka. the number of times a word appears in a document, divided by the total number of words in that document; the second term is the Inverse Document Frequency (IDF), computed as the logarithm of the number of the documents in the corpus divided by the number of documents where the specific term appears.
# 
# **TF: Term Frequency**, which measures how frequently a term occurs in a document. Since every document is different in length, it is possible that a term would appear much more times in long documents than shorter ones. Thus, the term frequency is often divided by the document length (aka. the total number of terms in the document) as a way of normalization: 
# 
# *TF(t) = (Number of times term t appears in a document) / (Total number of terms in the document).*
# 
# **IDF: Inverse Document Frequency**, which measures how important a term is. While computing TF, all terms are considered equally important. However it is known that certain terms, such as "is", "of", and "that", may appear a lot of times but have little importance. Thus we need to weigh down the frequent terms while scale up the rare ones, by computing the following: 
# 
# *IDF(t) = log_e(Total number of documents / Number of documents with term t in it).*
# 
# See below for a simple example.
# 
# **Example:**
# 
# Consider a document containing 100 words wherein the word cat appears 3 times. 
# 
# The term frequency (i.e., tf) for cat is then (3 / 100) = 0.03. Now, assume we have 10 million documents and the word cat appears in one thousand of these. Then, the inverse document frequency (i.e., idf) is calculated as log(10,000,000 / 1,000) = 4. Thus, the Tf-idf weight is the product of these quantities: 0.03 * 4 = 0.12.
# 

# In[33]:


from sklearn.feature_extraction.text import TfidfTransformer


# In[34]:


tfidf_transformer = TfidfTransformer().fit(messages_bow)
#to get an idea what it looks like
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)


# We'll go ahead and check what is the IDF (inverse document frequency) of the word "u" and of word "university"?

# In[35]:


print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])


# In[36]:


#To transform the entire bag-of-words corpus into TF-IDF corpus at once:
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)


# ## Training a model

# We'll be using scikit-learn here, choosing the [Naive Bayes](http://en.wikipedia.org/wiki/Naive_Bayes_classifier) classifier to start with:

# In[37]:


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages['label'])


# Let's try classifying our single random message 

# In[38]:


print('predicted:', spam_detect_model.predict(tfidf4)[0])
print('expected:', messages.label[3])


# ## Model Evaluation

# In[39]:


all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)


# In[40]:


from sklearn.metrics import classification_report
print (classification_report(messages['label'], all_predictions))


# In the above "evaluation",we evaluated accuracy on the same data we used for training. A proper way is to split the data into a training/test set, where the model only ever sees the **training data** during its model fitting and parameter tuning. The **test data** is never used in any way. This is then our final evaluation on test data is representative of true predictive performance.

# ## Train Test Split

# In[41]:


from sklearn.model_selection import train_test_split
msg_train, msg_test, label_train, label_test = train_test_split(messages['message'], messages['label'], test_size=0.2)


# ## Creating a Data Pipeline

# In[42]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[43]:


pipeline.fit(msg_train,label_train)


# In[44]:


predictions = pipeline.predict(msg_test)


# In[45]:


print(classification_report(predictions,label_test))


# In[ ]:




