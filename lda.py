import pandas as pd
import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

def sent_to_words(sentences):
    for sentence in sentences:
        # deacc=True removes punctuations
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc))
             if word not in stop_words] for doc in texts]


rootdir ='data'
data =[]
# Creates a list containing 20 lists, each of 1000 items, all set to 0
# Matrix = [[0 for x in range(20)] for y in range(1000)] 
Matrix = [[0,0] for y in range(1000)] 

i=0
for topic in os.listdir(rootdir):
    list =[]
    for subdir, dirs, files in os.walk(os.path.join(rootdir, topic)):
        file =files[0]
#         for file in files:
        if (topic == "sci.med"):
            with open(os.path.join(rootdir, topic,file)) as f:
                lines = f.readlines()
                count =0
                for line in lines :
                    if(line != "\n" ):
                        count= count +1
                    else:
                        content = lines[count:]
#                         print ('contents',count,content)
                        list.append(content)
                        break
                        # print ('file:',file)
                        filepath = subdir + os.sep + file
    Matrix[i]=[topic,list]
    i=i+1

# frame = pd.concat(list, axis=0, ignore_index=True)
# print ('frame',frame)
print ('Matrix',Matrix)
data_frame = pd.DataFrame(Matrix, columns=['t', 'c'])
print ('data_frame',Matrix)


#             pd.read
# Read data into papers
# papers = pd.from('./data/NIPS Papers/papers.csv')

#TODO read file
# https://github.com/kapadias/mediumposts/blob/master/natural_language_processing/topic_modeling/notebooks/Introduction%20to%20Topic%20Modeling.ipynb
