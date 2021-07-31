#!/usr/bin/env python
# coding: utf-8

# ## Sentiment Analysis on Amazon Fine Food Reviews:
# 
# ### Introduction:
# In this project, I produced a sentiment analysis covering over 500,000 reviews from an Amazon Fine Food Review dataset. I classified all positive and negative customer reviews and then created word clouds, plotly visualizations, and a text classification model to display my analysis further.
# 
# ### Data : 
# For this project, I used the Amazon Fine Food Review [dataset](https://www.kaggle.com/snap/amazon-fine-food-reviews?select=Reviews.csv) found on Kaggle.

# In[2]:


# @hidden_cell
# Installs
get_ipython().system(' pip install plotly')
get_ipython().system(' pip install cufflinks')
get_ipython().system(' pip install seaborn')
get_ipython().system(' pip install wordcloud')
get_ipython().system(' pip install numpy')
get_ipython().system(' pip install nltk')

# Imports
import os, types
import pandas as pd
import numpy as np
import seaborn as sns
import nltk
#nltk.download('stopwords')
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import cufflinks as cf
cf.go_offline()

import plotly.express as px
import re
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

def __iter__(self): return 0

print("All imports installed...!")


# In[3]:


# @hidden_cell
from botocore.client import Config
import ibm_boto3

def __iter__(self): return 0


# The following code accesses a file in your IBM Cloud Object Storage. It includes your credentials.
# You might want to remove those credentials before you share the notebook.

if os.environ.get('RUNTIME_ENV_LOCATION_TYPE') == 'external':
    endpoint_31aa68bccd9348ca9eb616277dc06f2f = 'https://s3.us.cloud-object-storage.appdomain.cloud'
else:
    endpoint_31aa68bccd9348ca9eb616277dc06f2f = 'https://s3.private.us.cloud-object-storage.appdomain.cloud'

client_31aa68bccd9348ca9eb616277dc06f2f = ibm_boto3.client(service_name='s3',
    ibm_api_key_id='o1J0uCmz8Yd1oNnOAe_qAVom7E51F7HQxj1GqNLDISiA',
    ibm_auth_endpoint="https://iam.cloud.ibm.com/oidc/token",
    config=Config(signature_version='oauth'),
    endpoint_url=endpoint_31aa68bccd9348ca9eb616277dc06f2f)

body = client_31aa68bccd9348ca9eb616277dc06f2f.get_object(Bucket='dataanalystportfolioprojects-donotdelete-pr-iyxmgghnf4lomv',Key='Reviews.csv')['Body']
# add missing __iter__ method, so pandas accepts body as file-like object
if not hasattr(body, "__iter__"): body.__iter__ = types.MethodType( __iter__, body )

df_data_2 = pd.read_csv(body)
#df_data_2.head()

#df_data_1 = pd.read_csv('Reviews.csv')

amazon = df_data_2

amazon.head()


# In[16]:


amazon = df_data_2

amazon.head()


# ### Methodology: 
# 
# To prepare for this analysis, I visualized the product scores from the dataset in a histogram using the plotly library.

# In[5]:


# Visualizing Product Scores - Histogram

fig = px.histogram(amazon, x="Score")
fig.update_layout(title_text = "Product Score")
fig.show()


# From the blue histogram, we can see more positive customer ratings than negative. Therefore, the majority of Amazon’s product reviews are positive.

# #### Methodology: 
# 
# Next, I created a word cloud to show the most frequently used words in the text (review) column. Before starting, I checked for any null values and used natural language processing such as NLTK stopwords before generating my word cloud.

# In[4]:


amazon.isna().sum()


# The above code shows that column 'Text' doesn't have any null values.

# ### Review Word Cloud 

# In[5]:


text = " ".join(review for review in amazon.Text)

# Removing errors in Text column
stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])

wordcloud = WordCloud(width = 3000, height = 2000, random_state = 1, stopwords=stopwords, background_color = "white", colormap = 'Set1', collocations = False).generate(text)

plot_cloud(wordcloud)


# #### Methodology: 
# 
# Next, I added a sentiment column by classifying only positive and negative reviews using the dataset's 'Score' column. For this sentiment, I categorized all positive reviews as scores > 3, negative for scores < 3, and dropped all neutral scores, which  = 3. Note, the sentiment column will later be used as training data for the sentiment classification model. 

# In[6]:


amazon = amazon[amazon.Score != 3]

# Postive = 1 
# Negative = -1

amazon ["Sentiment"] = amazon["Score"].apply(lambda x: -1 if x < 3 else +1)

amazon.head()


# In[7]:


amazon.dtypes


# #### Methodology: 
# 
# After building the sentiment column, I also created word clouds to display the most frequently used words for both positive and negative product reviews, respectfully. In addition, I made a product sentiment histogram to show the distribution of reviews with sentiment across the dataset.

# In[8]:


# Postive Word Cloud

positive = amazon[amazon["Sentiment"] == 1]

text = " ".join(review for review in positive.Text)

text = text.replace('\n', "")

stopwords = set(STOPWORDS)
stopwords.update(["br", "href"])

wordcloud.postive = WordCloud(width = 3000, height = 2000, random_state = 1, stopwords=stopwords, background_color = "black", colormap = 'Set2', collocations = False).generate(text)

plot_cloud(wordcloud.postive)


# In[9]:


# Negative Word Cloud

negative = amazon[amazon["Sentiment"] == -1]

text = " ".join(review for review in negative.Text)

text = text.replace('\n', "")

stopwords = set(STOPWORDS)
stopwords.update(["good", "great", "br", "href"])

wordcloud.negative = WordCloud(width = 3000, height = 2000, random_state = 1, stopwords=stopwords, background_color = "black", colormap = 'rainbow', collocations = False).generate(text)

plot_cloud(wordcloud.negative)


# ## Product Sentiment Histogram

# In[7]:


amazon ["Sentiment_Rate"] = amazon["Sentiment"].apply(lambda x: "Negative" if x == -1 else "Positive")

fig = px.histogram(amazon, x = "Sentiment_Rate")

fig.update_traces(marker_color = 'orange', marker_line_width=1.5)

fig.update_layout(title_text = "Product Sentiment")

fig.show()


# From the orange histogram, we can see that the product sentiment is more positive than negative. 

# #### Methodology: 
# 
# Finally, I created a text classification model to train and establish the accuracy of my data. I start by pre-processing the textual data using NLTK to remove special characters, lowercasing text, and stopwords. Then, I test the accuracy of the sentiment model by performing the Multi Nominal Naive Bayes Classification function using the scikit-learn library. 

# In[11]:


amazon.Summary = amazon['Summary'].str.replace('[^\w\s]','')

amazon.head()


# In[12]:


sentiment_df = amazon[["Summary", "Sentiment"]]

sentiment_df.head()


# Data Pre-Processing

# In[13]:


df = sentiment_df

df["Summary"] = df["Summary"].astype(str)

# Change to lowercasing for all text reviews in 'Summary'

df["Summary"] = df["Summary"].apply(lambda x: " ".join(x.lower() for x in x.split()))

df["Summary"][2]


# In[14]:


stop = set(stopwords)

df["Summary"] = df["Summary"].apply(lambda x: " ".join(x for x in x.split() if x not in stop ))

df["Summary"][2]


# In[15]:


cv = CountVectorizer(token_pattern=r'\b\w+\b')
text_counts = cv.fit_transform(df["Summary"])

X_train, X_test, y_train, y_test = train_test_split(
    text_counts, df["Sentiment"], test_size=0.3, random_state=1)

# Multinomial Naive Bayes Model
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)


print("Multinomial Naive Bayes Accuracy:",metrics.accuracy_score(y_test, predicted))


# As a result, the overall classification rate has an approx. 90.5% accuracy!
