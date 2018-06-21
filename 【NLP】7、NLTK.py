# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 10:08:56 2018

@author: wenzhe.jwz
"""

import nltk
import jieba

raw=open('Word2vec.txt',encoding='utf-8').read()
text=nltk.text.Text(jieba.lcut(raw))

###1、Text类介绍
print (text.concordance('阿里巴巴',lines = 1,width = 20))
print (text.common_contexts(['阿里巴巴','小鹏']))
text.collocations()
text.dispersion_plot([u'校园',u'大学'])
text.dispersion_plot(['阿里巴巴'])
text.vocab()
text.similar('阿里巴巴')
text.count('阿里巴巴')
text.idf('阿里巴巴')
###2、对文档用词进行分布统计
nltk.FreqDist.B(raw)
fdist = nltk.FreqDist(nltk.word_tokenize(raw))
fdist.plot(30,cumulative = True)
###3、nltk自带的语料库
nltk.corpus.fileids()

porter = nltk.PorterStemmer()
porter.stem('begin')

lema = nltk.WordNetLemmatizer()
lema.lemmatize('women')

####################################################NLTK学习之二：建构词性标注器
nltk.help.brown_tagset()
###1、使用NLTK对英文进行词性标注
#最实用标注器
import nltk
from nltk.corpus import stopwords  ##去除停用词
sent = 'i am going to ,beijing ,tomorrow'
tokens = nltk.word_tokenize(sent)
tokens = [w for w in tokens if(w not in stopwords.words('english'))]
tokens = [w for w in tokens if(w not in ['.',',','!','?'])]
taged_sent = nltk.pos_tag(tokens) ##词性标注

###2 标注器
#默认标注器
import nltk
from nltk.corpus import brown

default_tagger = nltk.DefaultTagger('NN')
sents = 'i am going to beijing today'
print (default_tagger.tag(sents))
tagged_sents = brown.tagged_sents(categories='news')
print (default_tagger.evaluate(tagged_sents)) #0.131304

#基于规则的标注器
from nltk.corpus import brown

pattern =[
    (r'.*ing$','VBG'),
    (r'.*ed$','VBD'),
    (r'.*es$','VBZ'),
    (r'.*\'s$','NN$'),
    (r'.*s$','NNS'),
    (r'.*', 'NN')  #未匹配的仍标注为NN
]
sents = 'I am going to Beijing.'
tagger = nltk.RegexpTagger(pattern)
print(tagger.tag(nltk.word_tokenize(sents)))

tagged_sents = brown.tagged_sents(categories='news')
print (tagger.evaluate(tagged_sents)) #0.1875

#基于查表的标注器
import nltk
from nltk.corpus import brown

fdist = nltk.FreqDist(brown.words(categories='news'))
ommon_word = fdist.most_common(10000)
cfdist = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
table= dict((word, cfdist[word].max()) for (word, _) in common_word)

uni_tagger = nltk.UnigramTagger(model=table,backoff=nltk.DefaultTagger('NN'))
print (uni_tagger.evaluate(tagged_sents)) #0.5817


###3 训练N-gram标注器
#一般N-gram标注
import nltk
from nltk.corpus import brown

brown_tagged_sents = brown.tagged_sents(categories='news')
train_num = int(len(brown_tagged_sents) * 0.9)
x_train =  brown_tagged_sents[0:train_num]
x_test =   brown_tagged_sents[train_num:]
tagger = nltk.UnigramTagger(train=x_train)
print (tagger.evaluate(x_test))  #0.81

#组合标注器
import nltk
from nltk.corpus import brown
pattern =[
    (r'.*ing$','VBG'),
    (r'.*ed$','VBD'),
    (r'.*es$','VBZ'),
    (r'.*\'s$','NN$'),
    (r'.*s$','NNS'),
    (r'.*', 'NN')  #未匹配的仍标注为NN
]
brown_tagged_sents = brown.tagged_sents(categories='news')
train_num = int(len(brown_tagged_sents) * 0.9)
x_train =  brown_tagged_sents[0:train_num]
x_test =   brown_tagged_sents[train_num:]

t0 = nltk.RegexpTagger(pattern)
t1 = nltk.UnigramTagger(x_train, backoff=t0)
t2 = nltk.BigramTagger(x_train, backoff=t1)
print (t2.evaluate(x_test))  #0.863

#基于Unigram训练一个中文词性标注器，语料使用网上可以下载得到的人民日报98年1月的标注资料
import nltk
import json

lines = open('词性标注人民日报.txt',encoding='utf-8').readlines()
all_tagged_sents = []

for line in lines:
    sent = line.split()
    tagged_sent = []
    for item in sent:
        pair = nltk.str2tuple(item)
        tagged_sent.append(pair)

    if len(tagged_sent)>0:
        all_tagged_sents.append(tagged_sent)

train_size = int(len(all_tagged_sents)*0.8)
x_train = all_tagged_sents[:train_size]
x_test = all_tagged_sents[train_size:]

tagger = nltk.UnigramTagger(train=x_train,backoff=nltk.DefaultTagger('n'))

tokens = nltk.word_tokenize(u'我 认为 不丹 的 被动 卷入 不 构成 此次 对峙 的 主要 因素。')
tagged = tagger.tag(tokens)
#["我", "R"], ["认为", "V"], ["不丹", "n"], ["的", "U"], ["被动", "A"], ["卷入", "V"], ["不", "D"], ["构成", "V"], ["此次", "R"], ["对峙", "V"], ["的", "U"], ["主要", "B"], ["因素。", "n"]
print (tagger.evaluate(x_test)) #0.871

####################################################NLTK学习之三：文本分类与构建基于分类的词性标注器
##1、文本分类示例
import random
import nltk
from nltk.corpus import movie_reviews

docs = [(list(movie_reviews.words(fileid)),category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

random.shuffle(docs)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
most_comment_word = [word for (word,_) in all_words.most_common(2000)]

def doc_feature(doc):
    doc_words = set(doc)
    feature = {}
    for word in most_comment_word:
        feature[word] = (word in doc_words)
    return feature

train_set = nltk.apply_features(doc_feature,docs[:100])
test_set = nltk.apply_features(doc_feature,docs[100:])

classifier = nltk.NaiveBayesClassifier.train(train_set)
print (nltk.classify.accuracy(classifier,test_set)) #0.735
classifier.show_most_informative_features()

##2、基于上下文的词性标注器

import nltk
from nltk.corpus import brown

def pos_feature_use_hist(sentence,i,history):
    features = {
        'suffix-1': sentence[i][-1:],
        'suffix-2': sentence[i][-2:],
        'suffix-3': sentence[i][-3:],
        'pre-word': 'START',
        'prev-tag': 'START'
    }
    if i>0:
        features['prev-word'] = sentence[i-1],
        features['prev-tag'] =  history[i-1]
    return features

class ContextPosTagger(nltk.TaggerI):
    def __init__(self,train):
        train_set = []
        for tagged_sent in train:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i,(word,tag) in enumerate(tagged_sent):
                features = pos_feature_use_hist(untagged_sent,i,history)
                train_set.append((features,tag))
                history.append(tag)
        print (train_set[:10])
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

    def tag(self,sent):
        history = []
        for i,word in enumerate(sent):
            features = pos_feature_use_hist(sent,i,history)
            tag = self.classifier.classify(features)
            history.append(tag)
        return zip(sent,history)

tagged_sents = brown.tagged_sents(categories='news')
size = int(len(tagged_sents)*0.8)z
train_sents,test_sents = tagged_sents[0:size],tagged_sents[size:]

#tagger = nltk.ClassifierBasedPOSTagger(train=train_sents)  # 0.881
tagger = ContextPosTagger(train_sents)  #0.78
tagger.classifier.show_most_informative_features()
print (tagger.evaluate(test_sents))

##混淆矩阵
import nltk
gold = [1,2,3,4]
test = [1,3,2,4]
print (nltk.ConfusionMatrix(gold,test))



####################################################NLTK学习之四：文本信息抽取

import nltk

sent = sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),("dog", "NN"), ("barked", "VBD"), ("at", "IN"),  ("the", "DT"), ("cat", "NN")]

grammer = 'AA:{<DT>*<JJ>*<NN>+}'
cp = nltk.RegexpParser(grammer)
tree = cp.parse(sent)

print (tree)
tree.draw()

import nltk

grammar = r"""
NP: {<DT|JJ|NN.*>+} 
PP: {<IN><NP>} 
VP: {<VB.*><NP|PP|CLAUSE>+$}
CLAUSE: {<NP><VP>}
"""
cp = nltk.RegexpParser(grammar，loop=2)
sentence = [("Mary", "NN"), ("saw", "VBD"), ("the", "DT"), ("cat", "NN"),("sit", "VB"), ("on", "IN"), ("the", "DT"), ("mat", "NN")]

cp.parse(sentence)

##
import re
import nltk
def open_ie():
    PR = re.compile(r'.*\president\b')
    for doc in nltk.corpus.ieer.parsed_docs():
        for rel in nltk.sem.extract_rels('PER', 'ORG', doc, corpus='ieer', pattern=PR):
            return nltk.sem.rtuple(rel)

print (open_ie())











