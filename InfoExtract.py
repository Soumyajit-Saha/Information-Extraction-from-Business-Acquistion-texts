from os import listdir
from os.path import isfile, join
import nltk
import re
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import wordnet as wn
import copy
import pandas as pd
from city_state import city_to_state_dict
import sys
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from nltk.chunk import conlltags2tree, tree2conlltags
from sklearn.ensemble import RandomForestClassifier
import spacy
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# import en_core_web_sm

import pickle
import random
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('popular')
# spacy.download('en_core_web_sm')

def vectorize_train_data(train_data):
    # Initialize vectorizer
    v = DictVectorizer(sparse=False)
    # We fit_transform in featurized train data (first transform the pandas dataframe into a list of dictionaries key = column name, value = entry in row)
    X_train = v.fit_transform(train_data.to_dict('records'))
    # We use the already fitted transform to transform the test data acccoring to the train data fit. (We also convert it into a list of dictionaries)
    # X_test=v.transform(test_data.to_dict('records'))
    # return vectorized train and test data
    return X_train

def vectorize_test_data(train_data, test_data):
    # Initialize vectorizer
    v = DictVectorizer(sparse=False)
    # We fit_transform in featurized train data (first transform the pandas dataframe into a list of dictionaries key = column name, value = entry in row)
    X_train = v.fit_transform(train_data.to_dict('records'))
    # We use the already fitted transform to transform the test data acccoring to the train data fit. (We also convert it into a list of dictionaries)
    X_test=v.transform(test_data.to_dict('records'))
    # return vectorized train and test data
    return X_train,X_test


#def perceptron(train_data, train_labels):
#    # initializing model
#    per = Perceptron(verbose=10, n_jobs=-1, max_iter=40)
#    # fitting model with train data and train labels
#    per.fit(train_data, train_labels)#, classes)
#    # saving the model to disk
#    filename = 'perceptron_model.joblib'
#    pickle.dump(per, open(filename, 'wb'))
#    return per

def logistic(train_data, train_labels):
    # initializing model
    # logistic = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)
    logistic = LogisticRegression(tol = 0.1, random_state=69, solver='saga', verbose=1, n_jobs=-1, max_iter=100, penalty='l1') 
    #warm_start=True, random_state=0, max_iter=5
    # logistic = RandomForestClassifier(random_state=69, verbose=1, n_jobs=-1, criterion="entropy")
    # fitting model with train data and train labels
    logistic.fit(train_data, train_labels)
    # saving the model to disk
    #filename = 'logistic_regression_model.joblib'
    #pickle.dump(logistic, open(filename, 'wb'))
    # return trained model
    return logistic

def logistic2(train_data, train_labels):
    # initializing model
    logistic = LogisticRegression(tol = 0.1, random_state=69, solver='sag', verbose=1, n_jobs=-1, max_iter=100) #, penalty='elasticnet', l1_ratio=0.5) #warm_start=True, random_state=0, max_iter=5
    # logistic = RandomForestClassifier(random_state=69, verbose=1, n_jobs=-1, criterion="entropy")
    # fitting model with train data and train labels
    logistic.fit(train_data, train_labels)
    # saving the model to disk
    #filename = 'logistic_regression_model.joblib'
    #pickle.dump(logistic, open(filename, 'wb'))
    # return trained model
    return logistic


def logistic3(train_data, train_labels):
    # initializing model
    logistic = LogisticRegression(tol = 0.1, random_state=69, solver='saga', verbose=1, n_jobs=-1, max_iter=100, penalty='elasticnet', l1_ratio=0.5) #warm_start=True, random_state=0, max_iter=5
    # logistic = RandomForestClassifier(random_state=69, verbose=1, n_jobs=-1, criterion="entropy")
    # fitting model with train data and train labels
    logistic.fit(train_data, train_labels)
    # saving the model to disk
    #filename = 'logistic_regression_model.joblib'
    #pickle.dump(logistic, open(filename, 'wb'))
    # return trained model
    return logistic


def get_synonyms(word, pos):
  for synset in wn.synsets(word, pos=pos_to_wordnet_pos(pos)):
    for lemma in synset.lemmas():
        yield lemma.name()

def pos_to_wordnet_pos(penntag, returnNone=False):
   morphy_tag = {'NN':wn.NOUN, 'JJ':wn.ADJ,'VB':wn.VERB, 'RB':wn.ADV, 'JJR':wn.ADJ, 'JJS':wn.ADJ, 'NNP':wn.NOUN, 'NNS':wn.NOUN, 'RBR':wn.ADV, 'RBS':wn.ADV, 'VBD':wn.VERB, 'VBG':wn.VERB, 'VBN':wn.VERB, 'VBP':wn.VERB, 'VBZ':wn.VERB, 'WRB':wn.ADV}
   try:
       return morphy_tag[penntag[:2]]
   except:
        return None if returnNone else ''


# --------TRAINING--------


docpath='./development-docs/development-docs'
docfiles = [f for f in listdir(docpath) if isfile(join(docpath, f))]
keypath='./development-anskeys/development-anskeys'

doc_list=[]
key_list=[]

for i in docfiles:
    f_doc=open(join(docpath,i),"r")
    doc=f_doc.read()
    doc_list.append(doc)



    f_key=open(join(keypath,i+'.key'),"r")
    key=f_key.read()
    key_list.append(key)
    # print(doc)
    # print(key)
    # print()

f_doc.close()

# print(doc_list)
# print()
# print(key_list)

key_lines_list=[]
doc_words_list=[]

doc_sent_list=[]

for i,keys in enumerate(key_list):
    key_lines=keys.split('\n')
    key_lines = list(filter(("").__ne__, key_lines))
    key_lines_list.append(key_lines)

#print(key_lines_list)
# nlp=en_core_web_sm.load()

for i,docs in enumerate(doc_list):
    # doc_lines=docs.split('\n')
    # doc_lines = list(filter(("").__ne__, doc_lines))
    # doc_words=[]

    # doc_sent=sent_tokenize(docs)

    # for sent in doc_sent:
    # words=word_tokenize(docs)
    # doc_words.append(words)
    w=[]
    for word in word_tokenize(docs):
        if '-based' in word:
            word=word.split('-')
            w.extend(word_tokenize(word[0]))
            w.extend('-')
            w.extend(word_tokenize(word[1]))
        
        else:
            w.extend(word_tokenize(word))
    doc_words_list.append(w)
    doc_sent_list.append(sent_tokenize(docs))

    # print(doc_words_list[-1])

key_docs_dict={}

for i in range(len(key_lines_list)):
    doc_no=key_lines_list[i][0].split(' ')[1]
    key_docs_dict[doc_no]=key_lines_list[i][1:]
    key_docs_dict[doc_no].append(doc_words_list[i][0:])
    key_docs_dict[doc_no].append(doc_sent_list[i])


WORD=[]
POS=[]
CLASS=[]
WORD_AFTER1=[]
WORD_AFTER2=[]
WORD_BEFORE1=[]
WORD_BEFORE2=[]
POS_AFTER1=[]
POS_AFTER2=[]
POS_BEFORE2=[]
POS_BEFORE1=[]
BIO_BEFORE1=[]
BIO_BEFORE2=[]
BIO_AFTER1=[]
BIO_AFTER2=[]
ROOT_BEFORE1=[]
ROOT_BEFORE2=[]
ROOT_AFTER1=[]
ROOT_AFTER2=[]
CAP=[]
SYN=[]
ROOT=[]
DOC_NO=[]
BIO=[]
IS_NUM=[]
LEMMA=[]
LEMMA_AFTER1=[]
LEMMA_AFTER2=[]
LEMMA_BEFORE1=[]
LEMMA_BEFORE2=[]
doc_no = 1

error=[]

for key in key_docs_dict:

    if True:

        class_dict={'ACQUIRED':[], 'ACQBUS':[], 'ACQLOC':[], 'DLRAMT':[], 'PURCHASER':[], 'SELLER':[], 'STATUS':[]}
        index_list=list(class_dict)

        # print(key)
        # print()

        for j in key_docs_dict.get(key):
            # print(j)
            if type(j)==str:
                key_line=j.split(':')
                # print(key_line)
                if key_line[1]!=' ---':
                    values=key_line[1].split('"')
                    values=list(filter(('').__ne__, values))
                    values=list(filter((' ').__ne__, values))
                    values=list(filter((' / ').__ne__, values))
                    # print(values)
                    for k in values:
                        # print(k,word_tokenize(k))
                        word_list=word_tokenize(k)
                        w=[]
                        for word in word_list:
                            if '-based' in word:
                                word=word.split('-')
                                w.extend(word_tokenize(word[0]))
                                w.extend('-')
                                w.extend(word_tokenize(word[1]))
                            else:
                                w.extend(word_tokenize(word))
                        class_dict[key_line[0]].append(w)
        # print(class_dict)
        # print()

        document=key_docs_dict[key][-2]

        document_sents=key_docs_dict[key][-1]


        document_POS_tag=[]
        BIO_tag=[]
        # for word in document:
        # document_POS_tag.extend(pos_tag(document))



        for sent in document_sents:
            sent=sent.strip('\n')
            sent=sent.strip('\t')
            w=[]
            for word in word_tokenize(sent):
                if '-based' in word:
                    word=word.split('-')
                    w.extend(word_tokenize(word[0]))
                    w.extend('-')
                    w.extend(word_tokenize(word[1]))
                else:
                    w.extend(word_tokenize(word))

            document_POS_tag.extend(pos_tag(w))

            pattern='NP: {<DT>?<JJ>*<NN>}'

            cp=nltk.RegexpParser(pattern)

            # print(pos_tag(w))


            cs=cp.parse(pos_tag(w))
            iob_tagged = tree2conlltags(cs)

            BIO_tag.extend(iob_tagged)


        # print(document_POS_tag)
        # print()
        # print(BIO_tag)
        # print()

        synonyms=[]
        roots=[]
        lemmas=[]
        # for sent in document:
        #     root=[]
        for word in document:
                # syns = wordnet.synsets(word)
                # if len(syns)==0:
                #     syno.append('U')
                # else:
                #     syno.append(syns[0].lemmas()[0].name())

            stemmer = SnowballStemmer("english")
            lemmatizer=WordNetLemmatizer()
            lemmas.append(lemmatizer.lemmatize(word))
            roots.append(stemmer.stem(word))

            # synonyms.append(syno)
            # roots.append(root)
        for word,tag in document_POS_tag:
            # syno=[]
            # for word, tag in sent:
            unique = sorted(set(synonym for synonym in get_synonyms(word,tag) if synonym != word))
            if unique:
                synonyms.append(unique[0])
            else:
                synonyms.append('')
        # synonyms.append(syno)

        # print(synonyms)
        # print()
        # print(roots)

        DOC_CLASS = copy.deepcopy(document)


        for ans_key in class_dict:


            if class_dict[ans_key]!=[]:

                flag=0

                for element in class_dict[ans_key]:

                    # print(element)


                    for word_no in range(len(document) - len(element) + 1):

                        # for word_no in range(len(sent) - len(element) + 1):
                        if ans_key=='ACQBUS':

                            if ' '.join(element).lower() in ' '.join(document[word_no : word_no + len(element)]).lower():

                                    # print(sent[word_no : word_no + len(class_dict[ans_key])])

                                flag=1

                                for ind in range(word_no, word_no + len(element)):

                                    DOC_CLASS[ind]=ans_key + '_class'


                        elif ans_key=='ACQLOC':

                            if ' '.join(element).lower() in ' '.join(document[word_no : word_no + len(element)]).lower() or (' '.join(document[word_no : word_no + len(element)]) in city_to_state_dict.keys() and ' '.join(element).lower() in city_to_state_dict[' '.join(document[word_no : word_no + len(element)])].lower() ):

                                    # print(sent[word_no : word_no + len(class_dict[ans_key])])

                                flag=1

                                for ind in range(word_no, word_no + len(element)):

                                    DOC_CLASS[ind]=ans_key + '_class'

                        else:

                            if ' '.join(element).lower() in ' '.join(document[word_no : word_no + len(element)]).lower():

                                    # print(sent[word_no : word_no + len(class_dict[ans_key])])

                                flag=1

                                for ind in range(word_no, word_no + len(element)):

                                    DOC_CLASS[ind]= ans_key + '_class'

                if flag==0:
                    # print(key)
                    error.append(key)
        # print()
        # print(document)
        # print()
        # print(DOC_CLASS)
        # print()
        # print(key_docs_dict[key][-1])
        # print()
        # print()
        # print(error)

        #----------CREATING FEATURES----------

        for word_no,word in enumerate(document):

            WORD.append(word)

            IS_NUM.append(1 if word.replace(".","",1).isdigit() else 0)


            BIO.append(BIO_tag[word_no][2])

            POS.append(document_POS_tag[word_no][1])

            SYN.append(synonyms[word_no])

            ROOT.append(roots[word_no])

            LEMMA.append(lemmas[word_no])

            if word_no >= 1:
                WORD_BEFORE1.append(document[word_no-1])
                POS_BEFORE1.append(document_POS_tag[word_no-1][1])
                BIO_BEFORE1.append(BIO_tag[word_no-1][2])
                ROOT_BEFORE1.append(roots[word_no-1])
                LEMMA_BEFORE1.append(lemmas[word_no-1])

            else:
                WORD_BEFORE1.append('PHI')
                POS_BEFORE1.append('PHI_POS')
                BIO_BEFORE1.append('PHI_BIO')
                ROOT_BEFORE1.append('PHI_ROOT')
                LEMMA_BEFORE1.append('PHI_LEMMA')

            if word_no >= 2:
                WORD_BEFORE2.append(document[word_no-2])
                POS_BEFORE2.append(document_POS_tag[word_no-2][1])
                BIO_BEFORE2.append(BIO_tag[word_no-2][2])
                ROOT_BEFORE2.append(roots[word_no-2])
                LEMMA_BEFORE2.append(lemmas[word_no-2])

            else:
                WORD_BEFORE2.append('PHI')
                POS_BEFORE2.append('PHI_POS')
                BIO_BEFORE2.append('PHI_BIO')
                ROOT_BEFORE2.append('PHY_ROOT')
                LEMMA_BEFORE2.append('PHY_LEMMA')


            if word_no <= len(document)-2:
                WORD_AFTER1.append(document[word_no+1])
                POS_AFTER1.append(document_POS_tag[word_no+1][1])
                BIO_AFTER1.append(BIO_tag[word_no+1][2])
                ROOT_AFTER1.append(roots[word_no+1])
                LEMMA_AFTER1.append(lemmas[word_no+1])

            else:
                WORD_AFTER1.append('OMEGA')
                POS_AFTER1.append('OMEGA_POS')
                BIO_AFTER1.append('OMEGA_BIO')
                ROOT_AFTER1.append('OMEGA_ROOT')
                LEMMA_AFTER1.append('OMEGA_LEMMA')


            if word_no <= len(document)-3:
                WORD_AFTER2.append(document[word_no+2])
                POS_AFTER2.append(document_POS_tag[word_no+2][1])
                BIO_AFTER2.append(BIO_tag[word_no+2][2])
                ROOT_AFTER2.append(roots[word_no+2])
                LEMMA_AFTER2.append(lemmas[word_no+2])

            else:
                WORD_AFTER2.append('OMEGA')
                POS_AFTER2.append('OMEGA_POS')
                BIO_AFTER2.append('OMEGA_BIO')
                ROOT_AFTER2.append('OMEGA_ROOT')
                LEMMA_AFTER2.append('OMEGA_LEMMA')

            if word[0].isupper():
                CAP.append(1)
            else:
                CAP.append(0)

            DOC_NO.append(key)


            if DOC_CLASS[word_no] in ['ACQUIRED_class', 'ACQBUS_class', 'ACQLOC_class', 'DLRAMT_class', 'PURCHASER_class', 'SELLER_class', 'STATUS_class']:
                CLASS.append(DOC_CLASS[word_no].replace('_class',''))
            else:
                CLASS.append('NO_CLASS')
        doc_no+=1
    # if len(document)!=len(document_POS_tag):
    #     print('ERROR')
    #     break


training_feature=pd.DataFrame(columns=['WORD','POS','ROOT','CAP','WORD-1','WORD-2','WORD+1','WORD+2','DOC_NO','POS-1','POS-2','POS+1','POS+2','BIO','BIO-1','BIO-2','BIO+1','BIO+2','ROOT-1','ROOT-2','ROOT+1','ROOT+2','CLASS'])
training_feature['WORD']=WORD
training_feature['POS']=POS
training_feature['BIO']=BIO
# training_feature['IS_NUM']=IS_NUM
training_feature['ROOT']=ROOT
# training_feature['LEMMA']=LEMMA
training_feature['CAP']=CAP
training_feature['WORD-1']=WORD_BEFORE1
training_feature['WORD-2']=WORD_BEFORE2
training_feature['WORD+1']=WORD_AFTER1
training_feature['WORD+2']=WORD_AFTER2
training_feature['POS-1']=POS_BEFORE1
training_feature['POS-2']=POS_BEFORE2
training_feature['POS+1']=POS_AFTER1
training_feature['POS+2']=POS_AFTER2
training_feature['BIO-1']=BIO_BEFORE1
training_feature['BIO-2']=BIO_BEFORE2
training_feature['BIO+1']=BIO_AFTER1
training_feature['BIO+2']=BIO_AFTER2
training_feature['ROOT-1']=ROOT_BEFORE1
training_feature['ROOT-2']=ROOT_BEFORE2
training_feature['ROOT+1']=ROOT_AFTER1
training_feature['ROOT+2']=ROOT_AFTER2
# training_feature['LEMMA-1']=LEMMA_BEFORE1
# training_feature['LEMMA-2']=LEMMA_BEFORE2
# training_feature['LEMMA+1']=LEMMA_AFTER1
# training_feature['LEMMA+2']=LEMMA_AFTER2
training_feature['DOC_NO']=DOC_NO
training_feature['CLASS']=CLASS

training_feature2=pd.DataFrame(columns=['WORD','POS','ROOT','CAP','LEMMA','WORD-1','WORD-2','WORD+1','WORD+2','DOC_NO','POS-1','POS-2','POS+1','POS+2','BIO','BIO-1','BIO-2','BIO+1','BIO+2','ROOT-1','ROOT-2','ROOT+1','ROOT+2','LEMMA-1','LEMMA-2','LEMMA+1','LEMMA+2','CLASS'])
training_feature2['WORD']=WORD
training_feature2['POS']=POS
training_feature2['BIO']=BIO
# training_feature['IS_NUM']=IS_NUM
training_feature2['ROOT']=ROOT
training_feature2['LEMMA']=LEMMA
training_feature2['CAP']=CAP
training_feature2['WORD-1']=WORD_BEFORE1
training_feature2['WORD-2']=WORD_BEFORE2
training_feature2['WORD+1']=WORD_AFTER1
training_feature2['WORD+2']=WORD_AFTER2
training_feature2['POS-1']=POS_BEFORE1
training_feature2['POS-2']=POS_BEFORE2
training_feature2['POS+1']=POS_AFTER1
training_feature2['POS+2']=POS_AFTER2
training_feature2['BIO-1']=BIO_BEFORE1
training_feature2['BIO-2']=BIO_BEFORE2
training_feature2['BIO+1']=BIO_AFTER1
training_feature2['BIO+2']=BIO_AFTER2
training_feature2['ROOT-1']=ROOT_BEFORE1
training_feature2['ROOT-2']=ROOT_BEFORE2
training_feature2['ROOT+1']=ROOT_AFTER1
training_feature2['ROOT+2']=ROOT_AFTER2
training_feature2['LEMMA-1']=LEMMA_BEFORE1
training_feature2['LEMMA-2']=LEMMA_BEFORE2
training_feature2['LEMMA+1']=LEMMA_AFTER1
training_feature2['LEMMA+2']=LEMMA_AFTER2
training_feature2['DOC_NO']=DOC_NO
training_feature2['CLASS']=CLASS

# print(training_feature)

training_feature.to_csv('train.csv', index=False)

# training_feature = training_feature.apply(LabelEncoder().fit_transform)

x=training_feature.iloc[:, :-1]
# categorical_columns=['WORD','POS','SYN','ROOT','WORD-1','WORD-2','WORD+1','WORD+2','POS-1','POS-2','POS+1','POS+2','SYN-1','SYN-2','SYN+1','SYN+2','ROOT-1','ROOT-2','ROOT+1','ROOT+2']
# x = pd.get_dummies(x, columns=categorical_columns, drop_first=True)

x2=training_feature2.iloc[:, :-1]

y=training_feature.iloc[:, -1]

vec_train_data=vectorize_train_data(x)

vec_train_data2=vectorize_train_data(x2)

model = logistic(vec_train_data, y)

model2 = logistic2(vec_train_data, y)

model3 = logistic3(vec_train_data, y)

model4 = logistic2(vec_train_data2, y)

# model5 = logistic(vec_train_data2, y)


#----------TESTING----------


file_test=open(sys.argv[1],"r")
test=file_test.read()

test_files=test.split('\n')


doc_no=1

with open(sys.argv[1] +".templates","w") as output_file:

    for files in test_files:
        if files=='':
            continue

        WORD=[]
        POS=[]
        CLASS=[]
        WORD_AFTER1=[]
        WORD_AFTER2=[]
        WORD_BEFORE1=[]
        WORD_BEFORE2=[]
        POS_AFTER1=[]
        POS_AFTER2=[]
        POS_BEFORE2=[]
        POS_BEFORE1=[]
        BIO_BEFORE1=[]
        BIO_BEFORE2=[]
        BIO_AFTER1=[]
        BIO_AFTER2=[]
        ROOT_BEFORE1=[]
        ROOT_BEFORE2=[]
        ROOT_AFTER1=[]
        ROOT_AFTER2=[]
        CAP=[]
        BIO=[]
        ROOT=[]
        DOC_NO=[]
        IS_NUM=[]
        LEMMA=[]
        LEMMA_BEFORE1=[]
        LEMMA_BEFORE2=[]
        LEMMA_AFTER1=[]
        LEMMA_AFTER2=[]
        file=open(files,"r")
        file_name=files.split('/')[-1]

        test_doc=file.read()
        file.close()

        w=[]
        for word in word_tokenize(test_doc):
            if '-based' in word:
                word=word.split('-')
                w.extend(word_tokenize(word[0]))
                w.extend('-')
                w.extend(word_tokenize(word[1]))
            else:
                w.extend(word_tokenize(word))

        sents=sent_tokenize(test_doc)

        document_POS_tag=[]
            # for word in document:
            # document_POS_tag.extend(pos_tag(document))
        BIO_tag=[]

        for sent in sents:
            sent=sent.strip('\n')
            sent=sent.strip('\t')
            w1=[]
            for word in word_tokenize(sent):
                if '-based' in word:
                    word=word.split('-')
                    w1.extend(word_tokenize(word[0]))
                    w1.extend('-')
                    w1.extend(word_tokenize(word[1]))
                else:
                    w1.extend(word_tokenize(word))

            document_POS_tag.extend(pos_tag(w1))

            pattern='NP: {<DT>?<JJ>*<NN>}'

            cp=nltk.RegexpParser(pattern)
            cs=cp.parse(pos_tag(w1))
            iob_tagged = tree2conlltags(cs)

            BIO_tag.extend(iob_tagged)

        synonyms=[]
        roots=[]
        lemmas=[]
        # for sent in document:
        #     root=[]
        for word in w:
                    # syns = wordnet.synsets(word)
                    # if len(syns)==0:
                    #     syno.append('U')
                    # else:
                    #     syno.append(syns[0].lemmas()[0].name())

            stemmer = SnowballStemmer("english")
            lemmatizer=WordNetLemmatizer()
            lemmas.append(lemmatizer.lemmatize(word))
            roots.append(stemmer.stem(word))

                # synonyms.append(syno)
                # roots.append(root)
        for word,tag in document_POS_tag:
            # syno=[]
            # for word, tag in sent:
            unique = sorted(set(synonym for synonym in get_synonyms(word,tag) if synonym != word))
            if unique:
                synonyms.append(unique[0])
            else:
                synonyms.append('')

        for word_no,word in enumerate(w):

                WORD.append(word)

                IS_NUM.append(1 if word.replace(".","",1).isdigit() else 0)

                POS.append(document_POS_tag[word_no][1])

                BIO.append(BIO_tag[word_no][2])


                ROOT.append(roots[word_no])
                LEMMA.append(lemmas[word_no])

                if word_no >= 1:
                    WORD_BEFORE1.append(w[word_no-1])
                    POS_BEFORE1.append(document_POS_tag[word_no-1][1])
                    BIO_BEFORE1.append(BIO_tag[word_no-1][2])
                    ROOT_BEFORE1.append(roots[word_no-1])
                    LEMMA_BEFORE1.append(lemmas[word_no-1])

                else:
                    WORD_BEFORE1.append('PHI')
                    POS_BEFORE1.append('PHI_POS')
                    BIO_BEFORE1.append('PHI_BIO')
                    ROOT_BEFORE1.append('PHI_ROOT')
                    LEMMA_BEFORE1.append('PHY_LEMMA')

                if word_no >= 2:
                    WORD_BEFORE2.append(w[word_no-2])
                    POS_BEFORE2.append(document_POS_tag[word_no-2][1])
                    BIO_BEFORE2.append(BIO_tag[word_no-2][2])
                    ROOT_BEFORE2.append(roots[word_no-2])
                    LEMMA_BEFORE2.append(lemmas[word_no-2])

                else:
                    WORD_BEFORE2.append('PHI')
                    POS_BEFORE2.append('PHI_POS')
                    BIO_BEFORE2.append('PHI_BIO')
                    ROOT_BEFORE2.append('PHY_ROOT')
                    LEMMA_BEFORE2.append('PHY_LEMMA')

                if word_no <= len(w)-2:
                    WORD_AFTER1.append(w[word_no+1])
                    POS_AFTER1.append(document_POS_tag[word_no+1][1])
                    BIO_AFTER1.append(BIO_tag[word_no+1][2])
                    ROOT_AFTER1.append(roots[word_no+1])
                    LEMMA_AFTER1.append(lemmas[word_no+1])

                else:
                    WORD_AFTER1.append('OMEGA')
                    POS_AFTER1.append('OMEGA_POS')
                    BIO_AFTER1.append('OMEGA_BIO')
                    ROOT_AFTER1.append('OMEGA_ROOT')
                    LEMMA_AFTER1.append('OMEGA_LEMMA')

                if word_no <= len(w)-3:
                    WORD_AFTER2.append(w[word_no+2])
                    POS_AFTER2.append(document_POS_tag[word_no+2][1])
                    BIO_AFTER2.append(BIO_tag[word_no+2][2])
                    ROOT_AFTER2.append(roots[word_no+2])
                    LEMMA_AFTER2.append(lemmas[word_no+2])


                else:
                    WORD_AFTER2.append('OMEGA')
                    POS_AFTER2.append('OMEGA_POS')
                    BIO_AFTER2.append('OMEGA_BIO')
                    ROOT_AFTER2.append('OMEGA_ROOT')
                    LEMMA_AFTER2.append('OMEGA_LEMMA')

                if word[0].isupper():
                    CAP.append(1)
                else:
                    CAP.append(0)

                DOC_NO.append(file_name)


        doc_no+=1
        # if len(document)!=len(document_POS_tag):
        #     print('ERROR')
        #     break


        testing_feature=pd.DataFrame(columns=['WORD','POS','ROOT','CAP','WORD-1','WORD-2','WORD+1','WORD+2','DOC_NO','POS-1','POS-2','POS+1','POS+2','BIO','BIO-1','BIO-2','BIO+1','BIO+2','ROOT-1','ROOT-2','ROOT+1','ROOT+2'])
        testing_feature['WORD']=WORD
        testing_feature['POS']=POS
        # testing_feature['IS_NUM']=IS_NUM
        testing_feature['BIO']=BIO
        testing_feature['ROOT']=ROOT
        # testing_feature['LEMMA']=LEMMA
        testing_feature['CAP']=CAP
        testing_feature['WORD-1']=WORD_BEFORE1
        testing_feature['WORD-2']=WORD_BEFORE2
        testing_feature['WORD+1']=WORD_AFTER1
        testing_feature['WORD+2']=WORD_AFTER2
        testing_feature['POS-1']=POS_BEFORE1
        testing_feature['POS-2']=POS_BEFORE2
        testing_feature['POS+1']=POS_AFTER1
        testing_feature['POS+2']=POS_AFTER2
        testing_feature['BIO-1']=BIO_BEFORE1
        testing_feature['BIO-2']=BIO_BEFORE2
        testing_feature['BIO+1']=BIO_AFTER1
        testing_feature['BIO+2']=BIO_AFTER2
        testing_feature['ROOT-1']=ROOT_BEFORE1
        testing_feature['ROOT-2']=ROOT_BEFORE2
        testing_feature['ROOT+1']=ROOT_AFTER1
        testing_feature['ROOT+2']=ROOT_AFTER2
        # testing_feature['LEMMA-1']=LEMMA_BEFORE1
        # testing_feature['LEMMA-2']=LEMMA_BEFORE2
        # testing_feature['LEMMA+1']=LEMMA_AFTER1
        # testing_feature['LEMMA+2']=LEMMA_AFTER2
        testing_feature['DOC_NO']=DOC_NO


        testing_feature2=pd.DataFrame(columns=['WORD','POS','ROOT','CAP','LEMMA','WORD-1','WORD-2','WORD+1','WORD+2','DOC_NO','POS-1','POS-2','POS+1','POS+2','BIO','BIO-1','BIO-2','BIO+1','BIO+2','ROOT-1','ROOT-2','ROOT+1','ROOT+2','LEMMA-1','LEMMA-2','LEMMA+1','LEMMA+2'])
        testing_feature2['WORD']=WORD
        testing_feature2['POS']=POS
        # testing_feature['IS_NUM']=IS_NUM
        testing_feature2['BIO']=BIO
        testing_feature2['ROOT']=ROOT
        testing_feature2['LEMMA']=LEMMA
        testing_feature2['CAP']=CAP
        testing_feature2['WORD-1']=WORD_BEFORE1
        testing_feature2['WORD-2']=WORD_BEFORE2
        testing_feature2['WORD+1']=WORD_AFTER1
        testing_feature2['WORD+2']=WORD_AFTER2
        testing_feature2['POS-1']=POS_BEFORE1
        testing_feature2['POS-2']=POS_BEFORE2
        testing_feature2['POS+1']=POS_AFTER1
        testing_feature2['POS+2']=POS_AFTER2
        testing_feature2['BIO-1']=BIO_BEFORE1
        testing_feature2['BIO-2']=BIO_BEFORE2
        testing_feature2['BIO+1']=BIO_AFTER1
        testing_feature2['BIO+2']=BIO_AFTER2
        testing_feature2['ROOT-1']=ROOT_BEFORE1
        testing_feature2['ROOT-2']=ROOT_BEFORE2
        testing_feature2['ROOT+1']=ROOT_AFTER1
        testing_feature2['ROOT+2']=ROOT_AFTER2
        testing_feature2['LEMMA-1']=LEMMA_BEFORE1
        testing_feature2['LEMMA-2']=LEMMA_BEFORE2
        testing_feature2['LEMMA+1']=LEMMA_AFTER1
        testing_feature2['LEMMA+2']=LEMMA_AFTER2
        testing_feature2['DOC_NO']=DOC_NO




        # print(testing_feature)


        testing_feature.to_csv('test.csv',index=False)

        # testing_feature = testing_feature.apply(LabelEncoder().fit_transform)

        # print(w)

        # print(word_tokenize(test_doc))

        # x=training_feature.iloc[:, :-1]
        # # categorical_columns=['WORD','POS','SYN','ROOT','WORD-1','WORD-2','WORD+1','WORD+2','POS-1','POS-2','POS+1','POS+2','SYN-1','SYN-2','SYN+1','SYN+2','ROOT-1','ROOT-2','ROOT+1','ROOT+2']
        # # x = pd.get_dummies(x, columns=categorical_columns, drop_first=True)
        # y=training_feature.iloc[:, -1]

        # print(x)
        # print(y)

        # svclassifier = SVC(kernel='rbf',coef0=2.0)
        # svclassifier.fit(x, y)

        x_test=testing_feature.iloc[:,:]
        # x_test=pd.get_dummies(x_test, columns=categorical_columns, drop_first=True)

        x_test2=testing_feature2.iloc[:,:]

        vec_train_data, vec_test_data = vectorize_test_data(x, x_test)

        vec_train_data2, vec_test_data2 = vectorize_test_data(x2, x_test2)

        y_pred = model.predict(vec_test_data)

        y_pred2 = model2.predict(vec_test_data)

        y_pred3 = model3.predict(vec_test_data)

        y_pred4 = model4.predict(vec_test_data2)

        # y_pred5 = model5.predict(vec_test_data2)



        # print(w)
        # print(len(w))
        # print(len(y_pred))
        # print(y_pred)
        acquired_list=[]
        purchaser_list=[]
        acqbus_list=[]
        acqloc_list=[]
        dlramt_list=[]
        purchaser_list=[]
        seller_list=[]
        status_list=[]

        classList=['ACQUIRED', 'ACQBUS','ACQLOC', 'DLRAMT', 'PURCHASER', 'SELLER', 'STATUS']
        for each in classList:
            indices = [i for i, x in enumerate(y_pred) if x==each]
            indices2 = [i for i, x in enumerate(y_pred2) if x==each]
            indices3 = [i for i, x in enumerate(y_pred3) if x==each]
            indices4 = [i for i, x in enumerate(y_pred4) if x==each]
            # indices5 = [i for i, x in enumerate(y_pred5) if x==each]

            if(each=='ACQUIRED'):
                c=0
                acq=[]
                for ind_no,ind in enumerate(indices):
                    if c==0:
                        acq.append(w[ind])
                        c=1
                    else:
                        if indices[ind_no]-indices[ind_no-1]==1 and ind_no!=len(indices)-1:
                            acq.append(w[ind])
                        elif indices[ind_no]-indices[ind_no-1]==1 and ind_no==len(indices)-1:
                            acq.append(w[ind])
                            acquired_list.append(acq)
                        elif indices[ind_no]-indices[ind_no-1]!=1 and ind_no!=len(indices)-1:
                            acquired_list.append(acq)
                            acq=[]
                            acq.append(w[ind])

                        else:
                            acquired_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                acquired_list.append(acq)

                acquired_list2=copy.deepcopy(acquired_list)

                acquired_list=[]

                temp_acq=[0]*len(acquired_list2)
                for j_no,j in enumerate(acquired_list2):
                    for t_no,t in enumerate(acquired_list2):
                        if t_no != j_no:
                            if ' '.join(acquired_list2[j_no]) in ' '.join(acquired_list2[t_no]) and temp_acq[t_no]==0:
                                temp_acq[j_no]=1

                for i,val in enumerate(temp_acq):
                    if val==0:
                        # print(i,acquired_list)
                        acquired_list.append(acquired_list2[i])
            elif(each=='ACQBUS'):
                c=0
                acq=[]
                for ind_no,ind in enumerate(indices4):
                    if c==0:
                        acq.append(w[ind])
                        c=1
                    else:
                        if indices4[ind_no]-indices4[ind_no-1]==1 and ind_no!=len(indices4)-1:
                            acq.append(w[ind])
                        elif indices4[ind_no]-indices4[ind_no-1]==1 and ind_no==len(indices4)-1:
                            acq.append(w[ind])
                            acqbus_list.append(acq)
                        elif indices4[ind_no]-indices4[ind_no-1]!=1 and ind_no!=len(indices4)-1:
                            acqbus_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                        else:
                            acqbus_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                acqbus_list.append(acq)

                acqbus_list2=copy.deepcopy(acqbus_list)

                acqbus_list=[]
                temp_acq=[0]*len(acqbus_list2)
                for j_no,j in enumerate(acqbus_list2):
                    for t_no,t in enumerate(acqbus_list2):
                        if t_no != j_no:
                            if ' '.join(acqbus_list2[j_no]).lower() in ' '.join(acqbus_list2[t_no]).lower() and temp_acq[t_no]==0:
                                temp_acq[j_no]=1

                for i,val in enumerate(temp_acq):
                    if val==0:
                        acqbus_list.append(acqbus_list2[i])
            elif(each=='ACQLOC'):
                c=0
                acq=[]
                for ind_no,ind in enumerate(indices3):
                    if c==0:
                        acq.append(w[ind])
                        c=1
                    else:
                        if indices3[ind_no]-indices3[ind_no-1]==1 and ind_no!=len(indices3)-1:
                            acq.append(w[ind])
                        elif indices3[ind_no]-indices3[ind_no-1]==1 and ind_no==len(indices3)-1:
                            acq.append(w[ind])
                            acqloc_list.append(acq)
                        elif indices3[ind_no]-indices3[ind_no-1]!=1 and ind_no!=len(indices3)-1:
                            acqloc_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                        else:
                            acqloc_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                acqloc_list.append(acq)

            
                acqloc_list2=copy.deepcopy(acqloc_list)
                

                acqloc_list=[]

                temp_acq=[0]*len(acqloc_list2)
                for j_no,j in enumerate(acqloc_list2):
                    for t_no,t in enumerate(acqloc_list2):
                        if t_no != j_no:
                            if ' '.join(acqloc_list2[j_no]) in ' '.join(acqloc_list2[t_no]) and temp_acq[t_no]==0:
                                temp_acq[j_no]=1

                for i,val in enumerate(temp_acq):
                    if val==0:
                        # print(acqloc_list)
                        acqloc_list.append(acqloc_list2[i])

                
                # acqloc_list=[acqloc_list[0]]
                        
            elif(each=='DLRAMT'):
                c=0
                acq=[]
                for ind_no,ind in enumerate(indices):
                    try:
                        if ind < len(y_pred)-1 and ind+1 not in indices and w[ind].replace(".","",1).isdigit() and (w[ind+1].lower()=='mln' or w[ind+1].lower()=='million' or w[ind+1].lower()=='bln' or w[ind+1].lower()=='billion' or w[ind+1].lower()=='tln' or w[ind+1].lower()=='trillion'):
                            indices.insert(ind_no+1,ind+1)
                        if ind < len(y_pred)-2 and ind+2 not in indices and w[ind].replace(".","",1).isdigit() and (w[ind+2].lower()=='dlrs' or w[ind+2].lower()=='dlr' or w[ind+2].lower()=='dollars' or w[ind+2].lower()=='dollar'):
                            indices.insert(ind_no+2,ind+2)
                        if ind < len(y_pred)-1 and ind+1 not in indices and (w[ind].lower()=='mln' or w[ind].lower()=='million' or w[ind].lower()=='bln' or w[ind].lower()=='billion' or w[ind].lower()=='tln' or w[ind].lower()=='trillion') and (w[ind+1].lower()=='dlrs' or w[ind+1].lower()=='dlr' or w[ind+1].lower()=='dollars' or w[ind+1].lower()=='dollar'):
                            indices.insert(ind_no+1,ind+1)
                        if ind >= 2 and (w[ind].lower()=='dlrs' or w[ind].lower()=='dlr' or w[ind].lower()=='dollars' or w[ind].lower()=='dollar') and w[ind-2].replace(".","",1).isdigit() and (w[ind-1].lower()=='mln' or w[ind-1].lower()=='million' or w[ind-1].lower()=='bln' or w[ind-1].lower()=='billion' or w[ind-1].lower()=='tln' or w[ind-1].lower()=='trillion') and ind-1 not in indices and ind-2 not in indices:
                            indices.insert(ind_no,ind-1)
                            indices.insert(ind_no,ind-2)
                        if ind >= 1 and (w[ind].lower()=='dlrs' or w[ind].lower()=='dlr' or w[ind].lower()=='dollars' or w[ind].lower()=='dollar') and w[ind-1].replace(".","",1).isdigit() and ind-1 not in indices:
                            indices.insert(ind_no,ind-1)
                        if ind >= 1 and ind-1 not in indices and (w[ind].lower()=='mln' or w[ind].lower()=='million' or w[ind].lower()=='bln' or w[ind].lower()=='billion' or w[ind].lower()=='tln' or w[ind].lower()=='trillion') and w[ind-1].replace(".","",1).isdigit():
                            indices.insert(ind_no,ind-1)
                    except IndexError:
                        var=10

                for ind_no,ind in enumerate(indices):
                    
                    if c==0:
                        acq.append(w[ind])
                        c=1
                    else:
                        if indices[ind_no]-indices[ind_no-1]==1 and ind_no!=len(indices)-1:
                            acq.append(w[ind])
                        elif indices[ind_no]-indices[ind_no-1]==1 and ind_no==len(indices)-1:
                            acq.append(w[ind])
                            dlramt_list.append(acq)
                        elif indices[ind_no]-indices[ind_no-1]!=1 and ind_no!=len(indices)-1:
                            dlramt_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                        else:
                            dlramt_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                dlramt_list.append(acq)

                # for word_no, word in enumerate(w):
                #     if word_no <= len(w)-3:
                #         if word.replace(".","",1).isdigit() and (w[word_no+1].lower()=='mln' or w[word_no+1].lower()=='million' or w[word_no+1].lower()=='bln' or w[word_no+1].lower()=='billion'or w[word_no+1].lower()=='tln' or w[word_no+1].lower()=='trillion') and (w[word_no+2].lower()=='dlrs' or w[word_no+2].lower()=='dollars'):
                #             dlramt_list.append([w[word_no],w[word_no+1],w[word_no+2]])
                
                # for word_no, word in enumerate(w):
                #     if word_no <= len(w)-2:
                #         if word.replace(".","",1).isdigit() and (w[word_no+1].lower()=='dlrs' or w[word_no+1].lower()=='dollars'):
                #             dlramt_list.append([w[word_no],w[word_no+1]])

                dlramt_list2=copy.deepcopy(dlramt_list)

                dlramt_list=[]
                temp_acq=[0]*len(dlramt_list2)
                for j_no,j in enumerate(dlramt_list2):
                    for t_no,t in enumerate(dlramt_list2):
                        if t_no != j_no:
                            if ' '.join(dlramt_list2[j_no]).lower() in ' '.join(dlramt_list2[t_no]).lower() and temp_acq[t_no]==0:
                                temp_acq[j_no]=1

                for i,val in enumerate(temp_acq):
                    if val==0:
                        dlramt_list.append(dlramt_list2[i])
                
                

            elif(each=='PURCHASER'):

                c=0
                acq=[]
                for ind_no,ind in enumerate(indices3):
                    if c==0:
                        acq.append(w[ind])
                        c=1
                    else:
                        if indices3[ind_no]-indices3[ind_no-1]==1 and ind_no!=len(indices3)-1:
                            acq.append(w[ind])
                        elif indices3[ind_no]-indices3[ind_no-1]==1 and ind_no==len(indices3)-1:
                            acq.append(w[ind])
                            purchaser_list.append(acq)
                        elif indices3[ind_no]-indices3[ind_no-1]!=1 and ind_no!=len(indices3)-1:
                            purchaser_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                        else:
                            purchaser_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                purchaser_list.append(acq)

                purchaser_list2=copy.deepcopy(purchaser_list)

                purchaser_list=[]

                temp_acq=[0]*len(purchaser_list2)
                for j_no,j in enumerate(purchaser_list2):
                    for t_no,t in enumerate(purchaser_list2):
                        if t_no != j_no:
                            if ' '.join(purchaser_list2[j_no]) in ' '.join(purchaser_list2[t_no]) and temp_acq[t_no]==0:
                                temp_acq[j_no]=1

                for i,val in enumerate(temp_acq):
                    if val==0:
                        purchaser_list.append(purchaser_list2[i])
            elif(each=='SELLER'):
                c=0
                acq=[]
                for ind_no,ind in enumerate(indices3):
                    if c==0:
                        acq.append(w[ind])
                        c=1
                    else:
                        if indices3[ind_no]-indices3[ind_no-1]==1 and ind_no!=len(indices3)-1:
                            acq.append(w[ind])
                        elif indices3[ind_no]-indices3[ind_no-1]==1 and ind_no==len(indices3)-1:
                            acq.append(w[ind])
                            seller_list.append(acq)
                        elif indices3[ind_no]-indices3[ind_no-1]!=1 and ind_no!=len(indices3)-1:

                            seller_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                        else:
                            seller_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                seller_list.append(acq)

                seller_list2=copy.deepcopy(seller_list)

                seller_list=[]
                temp_acq=[0]*len(seller_list2)
                for j_no,j in enumerate(seller_list2):
                    for t_no,t in enumerate(seller_list2):
                        if t_no != j_no:
                            if ' '.join(seller_list2[j_no]) in ' '.join(seller_list2[t_no]) and temp_acq[t_no]==0:
                                temp_acq[j_no]=1

                for i,val in enumerate(temp_acq):
                    if val==0:
                        seller_list.append(seller_list2[i])
            elif(each=='STATUS'):
                c=0
                acq=[]
                for ind_no,ind in enumerate(indices):
                    if c==0:
                        acq.append(w[ind])
                        c=1
                    else:
                        if indices[ind_no]-indices[ind_no-1]==1 and ind_no!=len(indices)-1:
                            acq.append(w[ind])
                        elif indices[ind_no]-indices[ind_no-1]==1 and ind_no==len(indices)-1:
                            acq.append(w[ind])
                            status_list.append(acq)
                        elif indices[ind_no]-indices[ind_no-1]!=1 and ind_no!=len(indices)-1:
                            status_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                        else:
                            status_list.append(acq)
                            acq=[]
                            acq.append(w[ind])
                status_list.append(acq)

                status_list2=copy.deepcopy(status_list)

                status_list=[]

                temp_acq=[0]*len(status_list2)
                for j_no,j in enumerate(status_list2):
                    for t_no,t in enumerate(status_list2):
                        if t_no != j_no:
                            
                            if ' '.join(status_list2[j_no]) in ' '.join(status_list2[t_no]) and temp_acq[t_no]==0:
                                temp_acq[j_no]=1

                for i,val in enumerate(temp_acq):
                    if val==0:
                        status_list.append(status_list2[i])

        # print("TEXT: ",file_name)
        # print("ACQUIRED: ", acquired_list)
        # print("ACQBUS: ", acqbus_list)
        # print("ACQLOC: ", acqloc_list)
        # print("DLRAMT: ", dlramt_list)
        # print("PURCHASER: ", purchaser_list)
        # print("SELLER: ", seller_list)
        # print("STATUS: ", status_list)

        # with open(file_name+".templates","w") as output_file:
        print("TEXT: "+file_name.split('.')[0],file=output_file)
        if acquired_list==[[]]:
            print('ACQUIRED: ---',file=output_file)
        else:
            for each in acquired_list:
                print('ACQUIRED: "', ' '.join(each) ,'"',sep='',file=output_file)
        if acqbus_list==[[]]:
            print('ACQBUS: ---',file=output_file)
        else:
            for each in acqbus_list:
                print('ACQBUS: "', ' '.join(each),'"',sep='',file=output_file)
        if acqloc_list==[[]]:
            print('ACQLOC: ---',file=output_file)
        else:
            for each in acqloc_list:
                print('ACQLOC: "', ' '.join(each),'"',sep='',file=output_file)
        if dlramt_list==[[]]:
            print('DLRAMT: ---',file=output_file)
        else:
            for each in dlramt_list:
                print('DLRAMT: "', ' '.join(each),'"',sep='',file=output_file)
        if purchaser_list==[[]]:
            print('PURCHASER: ---',file=output_file)
        else:
            for each in purchaser_list:
                print('PURCHASER: "', ' '.join(each),'"',sep='',file=output_file)
        if seller_list==[[]]:
            print('SELLER: ---',file=output_file)
        else:

            for each in seller_list:
                print('SELLER: "', ' '.join(each),'"',sep='',file=output_file)
        if status_list==[[]]:
            print('STATUS: ---',file=output_file)
        else:
            for each in status_list:
                print('STATUS: "', ' '.join(each),'"',sep='',file=output_file)
        print('\n',file=output_file)

output_file.close()