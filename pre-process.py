
#As we dont have any GPU we will be using google collab for GPU usage
#Mount at drive 
import sys
from google.colab import drive
from pathlib import Path
drive.mount("/content/drive", force_remount=True)

#needed packages to install
#!pip install texthero==1.0.5
!pip install gensim
!pip install openpyxl 
!pip install bnlp_toolkit
!pip install python-bidi
!pip install texthero
!pip install bangla-stemmer

#import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from wordcloud import WordCloud
from bnlp.corpus import stopwords, punctuations
import bnlp
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
from bnlp import BasicTokenizer,NLTKTokenizer
from bangla_stemmer.stemmer.stemmer import BanglaStemmer
import warnings
warnings.filterwarnings("ignore")

"""**Data Cleaning**"""

df = pd.read_excel('/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/dataset/multi class bangla social media comment.xlsx')
df.to_csv('/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/dataset/multi_class_bangla_social_media_comment.csv', encoding='utf-8', index=False)

df_plot=df.copy()
df.head()

df = df.rename(columns={'comment react number': 'comment_react_number'})
df.info()

#Details of each column

Category_counts=df.Category.value_counts()
Gender_counts=df.Gender.value_counts()
Comment_react_counts=df.comment_react_number.value_counts()
label_counts=df.label.value_counts()
print(Category_counts)
print("\n",Gender_counts)
print("\n",Comment_react_counts)
print("\n",label_counts)



df_abuse = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/dataset/new datasets/finaldata.csv', encoding='utf-8')
df_abuse.head()
df_abuse.label.value_counts()

col=['comment_react_number','Category','Gender']
df=df.drop(col,axis=1)
df.label.value_counts()
df.head()

df_abuse = df_abuse.rename(columns={'text': 'comment'})
df_abuse.head()

#df_abuse = df_abuse.rename(columns={'text': 'comment'})
df_abuse=df_abuse.drop(['target'],axis=1)
df_abuse.label.value_counts()

df_full=df.append(df_abuse)
df_full.label.value_counts()

#df_fake_sexual=df_full.loc[df_full.label == 'sexual'][:11000]
#df_fake_political=df_full.loc[df_full.label == 'Political'][:11000]
#df_religious=df_full.loc[df_full.label == 'religious']
#df_religious=df_religious.append([df_fake_political,df_fake_sexual])
#df_religious=df_religious.replace({'label': {"Political":"religious"}})
#df_religious.label.value_counts()

df_political=df_full.loc[df_full.label == 'Political'][:11000]
df_sexual=df_full.loc[df_full.label == 'sexual'][:11000]
df_not_bully=df_full.loc[df_full.label == 'not bully'][:11000]
df_religious=df_full.loc[df_full.label == 'religious']

df_political.label.value_counts()

#table
df=df_not_bully.append([df_religious,df_political,df_sexual])
df.label.value_counts()

from sklearn.utils import shuffle
df = shuffle(df)
df=df.reset_index(drop=True)
df.head()

df_copy=df.copy()

#preprocessing

def demoji(text):
	emoji_pattern = re.compile("["
		u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U00010000-\U0010ffff"
	                           "]+", flags=re.UNICODE)
	return(emoji_pattern.sub(r'', text)) 


def clean(text):
    text = re.sub('[%s]' % re.escape(punctuations), ' ', text)     #escape punctuation
    text = re.sub('\n', ' ', text)                                 #replace line break with space
    text = re.sub('\w*\d\w*', ' ', text)                           #ignore digits
    #text = re.sub('\xa0', ' ', text)                              
    return text

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[\u09E6-\u09FF]+', ' ', text)                  #remove bangla punctuations
    return text

df[u'text'] = df[u'comment'].astype(str)
df[u'text'] = df[u'text'].apply(lambda x:demoji(x))
df['text'] = df['text'].apply(lambda x: re.split('http:\/\/.*', str(x))) #remove urls
df["text"] = df['text'].apply(lambda x: clean(str(x)))                      
df['text'] = df['text'].apply(lambda x: remove_punct(x))

#remove special characters
spec_chars = ["!",'"',"।","#","%","&","'","(",")",
              "*","+",",","-",".","/",":",";","<",
              "=",">","?","@","[","\\","]","^","_",
              "`","{","|","}","~","–"]
for char in spec_chars:
    df['text'] = df['text'].str.replace(char, ' ') 
    df['text'] = df['text'].str.split().str.join(' ')         #remove whitespace

#checking......
df.text[2226]

#remove bangla stopwords

custom_stop_word_list=['আমার ','অথচ ','অথবা ','অনুযায়ী ','অনেক ','অনেকে ','অনেকেই ','অন্তত ','অন্য ','অবধি ','অবশ্য ','অর্থাত ','আই ','আগামী ','আগে ','আগেই ','আছে ','আজ ','আদ্যভাগে ',
                       'আপনার ','আপনারা ','আপনি ','আবার ','আসবে ','আমরা ',' আমাকে ','আমাদের ','আমার ','আমি ','আর ','আরও ','ইত্যাদি ','ইহা ','উচিত ','উত্তর ','উনি ','উপর ','উপরে ','এ ','এঁদের ','এঁরা ','এরা ',
                       'এই ','একই ','একটি ','একবার ','একে ','এক্ ','এখন ','এখনও ','এখানে ','এখানেই ','এটা ','এটাই ','এটি ','এত ','এতটাই ','এতে ','এদের ','এব ','এবং ','এবার ','এমন ','এমনকী ',
                       'এমনি ','এর ','এরা ','এল ','এস ','এসে ','ঐ ','ওঁদের ','ওঁর ','ওঁরা ','ওই ','ওকে ','ওখানে ','ওদের ','ওর ','ওরা ','কখনও ','কত ','কবে ','কমনে ','কয়েক ','কয়েকটি ','করছে ',
                       'করছেন ','করতে ',' করবে',' করবেন',' করলে ',' করলেন',' করা',' করাই',' করায়',' করার',' করি','করতে ','করিতে ','করিয়া ','করিয়ে ','করে ','করেই ','করেছিলেন ','করেছে ','করেছেন ','করেন ',
                       'কাউকে ','কাছ ','কাছে ','কাজ ','কাজে ','কারও ','কারণ ','কি ','কিংবা ','কিছু ','কিছুই ','হেতি ','কিন্তু ','ন্তু ','কী ','কে ','কেউ ','কেউই ','কেখা ','কেন ','কোটি ','কোন ','কোনও ',
                       'কোনো ','ক্ষেত্রে ','কয়েক ','খুব ','গিয়ে ','গিয়েছে ','গেছেন ','গিয়ে ','গুলি ','গেছে ','গেল ','গেলে ','গোটা ','চলে ','চান ','চায় ','চার ','চালু ','চেয়ে ','চেষ্টা ','ছাড়া ','ছাড়াও ','ছিল ','ছিলেন ','জন ',
                       'জনকে ','জনের ','জন্য ','জন্যওজে ','জানতে ','জানা ','জানানো ','জানায় ','জানিয়ে ','জানিয়েছে ','জ্নজন ','জন ','টা ','টি ','ঠিক ','তখন ','তত ','তথা ','তবু ','তবে ','তা ','তাঁকে ','তাঁদের ',
                       'তাঁর ','তোর ','তাঁরা ','তাঁহারা ','তাই ','যে ''তাও ','তাকে ','তাতে ','তাদের ','তার ','তারপর ','তারা ','তারৈ ','তাহলে ','তাহা ','তাহাতে ' ,'তাহার ','তিনঐ ','তিনি ','তিনিও ','তুমি ','তুলে ','তেমন ','তো ','তোমার ',
                       'থাকবে ','থাকবেন ','থাকা ','থাকায় ','থাকে ','থাকেন ','থেকে ','থেকেই ','থেকেও ','দিকে ','দিতে ','দিতাম','দিন ','দিয়ে ','দিয়েছে ','দিয়েছেন ','দিলেন ', 'দু ','দুই ','দুটি ','দুটো ','দেওয়া ','দেওয়ার ','দেওয়া ',
                       'দেখতে ','দেখা ','দেখে ','দেন ','দেয়া ','দেয় ','দ্বারা ','ধরা ','ধরে ','ধামার ','নতুন ','নাই ','নাকি ','নাগাদ ','নানা ','নিজে ','নিজেই ','নিজেদের ','নিজের ','নিতে ','নিয়ে ','নিয়ে ','নেই ','নেওয়া ','নেওয়ার ',
                       'নেওয়া ','নয় ','পক্ষে ','পর ','পরে ','পরেই ','পরেও ','পর্যন্ত ','পাওয়া ','পাচ ','পারি ','পারে ','পারেন ','পেয়ে ','পেয়্র্ ','প্রতি ','প্রথম ','প্রভৃতি ','প্রযন্ত ','প্রাথমিক ','প্রায় ','প্রায় ','ফলে ','ফিরে ','ফের ',
                       'বক্তব্য ','বদলে ','বন ','বরং ','বলতে ','বলছি ','বলল ','বললেন ','বলা ','বলে ','বলেছেন ','বলেন ','বসে ','বহু' ,'বাদে ','বার ','বিনা ','বিভিন্ন ','বিশেষ ','বিষয়টি ','বেশ ','বেশি ','ব্যবহার ','ব্যাপারে ','ভাবে ', 'ভাবেই ',
                       'মতো ','মতোই ','মধ্যভাগে ','মধ্যে ','মধ্যেই ','মধ্যেও ','মনে ','মাত্র ','মাধ্যমে ','মোট ','মোটেই ','যখন ','যত ','যতটা ','যথেষ্ট ','যদি ','যদিও ','যা ','যাঁর ','যাঁরা ','যাওয়া ','যাওয়ার ','যাওয়া ','যাকে ','যাচ্ছে ',
                       'যাতে ','যাদের ','যান ','যাবে ','যায় ','যার ','যারা ','যিনি ','অতএব ','যেখানে ','যেতে ','যেন ','যেমন ','রকম ','রয়েছে ','রাখা ','রেখে ','লক্ষ ','শুধু ','শুরু ','সঙ্গে ','সঙ্গেও ','সব ','সবার ','সবাইর ','সমস্ত ',
                       'সম্প্রতি ','সহ ','সহিত ','সবই ','সাধারণ ','সামনে ','সুতরাং ','সবাইর ','সে ','সেই ','সেখান ','সেখানে ','সেটা ', 'সেটাই ','সেটাও ','সেটি ','স্পষ্ট ','স্বয়ং ','হইতে ','হইবে ','হইয়া ','হওয়া ','হওয়ায় ','হওয়ার ','হচ্ছে ','হত ','হতে ',
                       'লেগেছে ','হতেই ','হন ','হইত ','হবে ','তিনি ','হবেন ','হয় ','হয়তো ','হয়নি ','হয়ে ','হয়েই ','হয়েছিল ','হয়েছে ','হয়েছেন ','হল ','হলে ','হলেই ','হলেও ','হলো ','হাজার ','হিসাবে ','হৈলে ','হোক ','হয় ']

#digits=['০ ','১ ','২ ','৩ ','৪ ','৫ ','৬ ','৭ ','৮ ','৯ ']

final_stopword_list = custom_stop_word_list 

pat = r'\b(?:{})\b'.format('|'.join(final_stopword_list))
df['text'] = df['text'].str.replace(pat, ' ')
df['text'] = df['text'].str.replace(r'\s+', ' ')

#checking......
df.text[2226]

#wordcloud to see importance of word for each label

regex = r"[\u0980-\u09FF]+"

# Start with one review:
df_NB = df[df['label']=="not bully"]
df_SE = df[df['label']=="sexual"]
df_RE = df[df['label']=="religious"]
df_TH = df[df['label']=="Political"]

text_NB = " ".join(review for review in df_NB.text)
text_SE = " ".join(review for review in df_SE.text)
text_RE = " ".join(review for review in df_RE.text)
text_TH = " ".join(review for review in df_TH.text)

fig, ax = plt.subplots(4, 1, figsize  = (30,30))
# Create and generate a word cloud image:
wordcloud_NB = WordCloud(font_path="/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/fonts/BenSenHandwriting.ttf",max_font_size=50,max_words=100,regexp=regex, background_color="white").generate(text_NB).to_file("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/wordcloud/not bully texts.png")
wordcloud_SE = WordCloud(font_path="/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/fonts/NikoshLight.ttf",max_font_size=50,max_words=100,regexp=regex, background_color="white").generate(text_SE).to_file("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/wordcloud/sexual texts.png")
wordcloud_RE = WordCloud(font_path="/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/fonts/MOHAO___.ttf",max_font_size=50,max_words=100,regexp=regex, background_color="white").generate(text_RE).to_file("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/wordcloud/religeous texts.png")
wordcloud_TH = WordCloud(font_path="/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/fonts/MOHAO___.ttf",max_font_size=50,max_words=100,regexp=regex, background_color="white").generate(text_TH).to_file("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/wordcloud/threat texts.png")


# Display the generated image:
ax[0].imshow(wordcloud_NB, interpolation='bilinear')
ax[0].set_title('Texts under Not Bully Class',fontsize=20)
ax[0].axis('off')

ax[1].imshow(wordcloud_SE, interpolation='bilinear')
ax[1].set_title('Texts Sexual Class',fontsize=20)
ax[1].axis('off')

ax[2].imshow(wordcloud_RE, interpolation='bilinear')
ax[2].set_title('Texts Religeous Class',fontsize=20)
ax[2].axis('off')

ax[3].imshow(wordcloud_TH, interpolation='bilinear')
ax[3].set_title('Texts Political Class',fontsize=20)
ax[3].axis('off')

df["clean_text"]=df["text"]

df.head()
df.head(0)

def stemming_text(corpus):
    stm = BanglaStemmer()
    return [' '.join([stm.stem(word) for word in review.split()]) for review in corpus]

df['text'] = stemming_text(df['text'])

df.head()



#Tokenization

b_token = BasicTokenizer()
df['tokenized_clean_text'] = df.apply(lambda row: b_token.tokenize(row['clean_text']), axis=1)
df['tokenized_stem_text'] = df.apply(lambda row: b_token.tokenize(row['text']), axis=1)

df['token_length'] = df.apply(lambda row: len(row['tokenized_clean_text']), axis=1)

df.head()

df.describe()

df.head()

df1=df.copy()
col=['comment']
df=df.drop(col,axis=1)
df.to_csv('/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/dataset/bangla_comments_tokenized.csv', index = False, header=True)
df.tail()



from bnlp import POS
df.tokenized_clean_text = df.tokenized_clean_text.astype(str)
def pos_tagging(doc):
  bn_pos = POS()
  model_path = "/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/bn_pos.pkl"
  doc = bn_pos.tag(model_path,doc)
  return doc

def get_postags(row):
    postags = pos_tagging(row["tokenized_clean_text"])
    list_classes = list()
    for  word in postags:
        list_classes.append(word[1])
    return list_classes

df["postags_list"] = df.apply(lambda row: get_postags(row), axis = 1)
df['postags_list'] = df.postags_list.apply(lambda x: [i for i in x if i != 'PU'])
df['postags_list'] = df.postags_list.apply(lambda x: [i for i in x if i != 'RDS'])
df.sample(10, random_state = 4)

df.shape

from collections import Counter
from functools import reduce

def find_no_class(count, class_name = ""):
    total = 0
    for key in count.keys():
      if key.startswith(class_name):
        total += count[key]        
    return total


def get_classes(row, grammatical_class = ""):
    count = Counter(row["postags_list"])
    try:
      return find_no_class(count, class_name = grammatical_class)/len(row["postags_list"])
    except ZeroDivisionError:
      return find_no_class(count, class_name = grammatical_class)
    

df["freqAdverbs"] = df.apply(lambda row: get_classes(row, "AMN"), axis = 1)
df["freqPreposition"] = df.apply(lambda row: get_classes(row, "PP"), axis = 1)
df["freqPronoun"] = df.apply(lambda row: get_classes(row, "PPR"), axis = 1)
df["freqVerbs"] = df.apply(lambda row: get_classes(row, "VM"), axis = 1)
df["freqAdjectives"] = df.apply(lambda row: get_classes(row, "JJ"), axis = 1)
df["freqNouns"] = df.apply(lambda row: get_classes(row, ("NC")), axis = 1)
df["freqEnglish"] = df.apply(lambda row: get_classes(row, ("RDF")), axis = 1)

df.tail()

df.head()

df.to_csv('/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/dataset/w_post_tag_bangla_comments_tokenized.csv', index = False, header=True)

#df1=df[df['tokenized_clean_text'].map(len) < 40]

df.shape

"""**Data Description and Visualization**"""

print("shape of data",df.shape)
print("Number of rows: "+str(df.shape[0]))
print("Number of columns: "+str(df.shape[1]))

df.describe().T

#countplot and piechart for label column
fig, ax=plt.subplots(1,2,figsize=(15,6))
_ = sns.countplot(x='label', data=df, ax=ax[0])
_ = df['label'].value_counts().plot.pie(autopct="%1.1f%%", ax=ax[1])

plt.savefig("/content/drive/MyDrive/Colab Notebooks/multi class bangla sentiment analysis/visualization/count plot and pie chart/label count plot and pie chart.png")

"""From the figure above we can see we got a small count of threat label so we need balance it with other classes and remove imbalancing from our dataframe."""

#parcentagewise null values
def missing_data_table(data):
    total = data.isnull().sum().sort_values(ascending=False)
    percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
missing_data_table(df)



df.loc[df["label"] == 'sexual', "freqNouns"].hist(alpha = 0.5);
df.loc[df["label"] == 'not bully', "freqNouns"].hist(alpha = 0.5);
df.loc[df["label"] == 'Political', "freqNouns"].hist(alpha = 0.5);
df.loc[df["label"] == 'religious', "freqNouns"].hist(alpha = 0.5);

df.loc[df["label"] == 'sexual', "freqAdjectives"].hist(alpha = 0.5);
df.loc[df["label"] == 'not bully', "freqAdjectives"].hist(alpha = 0.5);
df.loc[df["label"] == 'Political', "freqAdjectives"].hist(alpha = 0.5);
df.loc[df["label"] == 'religious', "freqAdjectives"].hist(alpha = 0.5);

df.loc[df["label"] == 'sexual', "freqVerbs"].hist(alpha = 0.5);
df.loc[df["label"] == 'not bully', "freqVerbs"].hist(alpha = 0.5);
df.loc[df["label"] == 'Political', "freqVerbs"].hist(alpha = 0.5);
df.loc[df["label"] == 'religious', "freqVerbs"].hist(alpha = 0.5);