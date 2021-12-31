import pandas as pd
import os
import nltk
import csv 
from pprint import pprint
import gensim.corpora as corpora
# Load the regular expression library
import re
# Import the wordcloud library
from wordcloud import WordCloud
import gensim
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])


# def writeToCsv(data):
#     header = ['topic','content']

#     with open('data.csv', 'w', encoding='UTF8') as f:
#         writer = csv.writer(f)

#         # write the header
#         writer.writerow(header)
#         print(data)

#         # write the data
#         writer.writerow(data)

def readData(rootdir):
    Matrix = [] # list of [topic, text]

    for topic in os.listdir(rootdir):
        for subdir, dirs, files in os.walk(os.path.join(rootdir, topic)):
            # file =files[0]
            for file in files:
                with open(os.path.join(rootdir, topic,file)) as f:
                    try:
                        lines = f.readlines()
                        count =0
                        for line in lines :
                            if(line != "\n" ):
                                count= count +1
                            else:
                                content = ' '.join(lines[count:])
                                Matrix.append([topic,content])
                                break
                    except:
                        print(f)
    dataFrame = pd.DataFrame(Matrix, columns=['topic', 'content'])
    return dataFrame

def sentenceToWords(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def removePunctuation(data):
     data['content']= data['content'].map(lambda x: re.sub('[,\.!?]', '', x))


def removeStopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def wordCloud(data):
    # Join the different processed titles together.
    long_string = ','.join(list(data['content'].values))
    print (long_string)

    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=1000, contour_width=3, contour_color='steelblue')

    # Generate a word cloud
    wordcloud.generate(long_string)

    # Visualize the word cloud
    wordcloud.to_image()

rootdir='data'
data = readData(rootdir)
removePunctuation(data)
# wordCloud(data)
data = data.content.values.tolist()
# print(data)
data_words = list(sentenceToWords(data))
# remove stop words
data_words = removeStopwords(data_words)
# print(data_words[:1][0][:30])
# Create Dictionary
id2word = corpora.Dictionary(data_words)
# Create Corpus
texts = data_words

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
# print(corpus[:1][0][:30])

# number of topics
num_topics = 20

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=num_topics)

# Print the Keyword in the 10 topics
# pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]


#             pd.read
# Read data into papers
# papers = pd.from('./data/NIPS Papers/papers.csv')

#TODO read file
# https://github.com/kapadias/mediumposts/blob/master/natural_language_processing/topic_modeling/notebooks/Introduction%20to%20Topic%20Modeling.ipynb
