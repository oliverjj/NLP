# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 14:19:16 2018

@author: wenzhe.jwz
"""

import jieba  
import networkx as nx  
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer  
  
def cut_sentence(sentence):  
    """ 
    分句 
    :param sentence: 
    :return: 
    """  
#    if not isinstance(sentence, unicode):  
#        sentence = sentence.decode('utf-8')  
    delimiters = frozenset('。！？')  
    buf = []  
    for ch in sentence:
        buf.append(ch)  
        if delimiters.__contains__(ch):  
            yield ''.join(buf)  
            buf = []  
    if buf:  
        yield ''.join(buf)  

def load_stopwords(path='D:/学习文档/机器学习/stopwords.txt'):  
    """ 
    加载停用词 
    :param path: 
    :return: 
    """  
    with open(path,encoding='utf-8') as f:  
        stopwords = filter(lambda x: x, map(lambda x: x.strip(), f.readlines()))
    stopwords = list(stopwords)
    map(lambda x :stopwords.append(x),[' ', '\t', '\n']) 
    return frozenset(stopwords)  

def cut_words(sentence):  
    """ 
    分词 
    :param sentence: 
    :return: 
    """  
    stopwords = load_stopwords()  
    return filter(lambda x: not stopwords.__contains__(x), jieba.cut(sentence))  
  
  
def get_abstract(content, size=3):  
    """ 
    利用textrank提取摘要 
    :param content: 
    :param size: 
    :return: 
    """  
    docs = list(cut_sentence(content))  
    tfidf_model = TfidfVectorizer(tokenizer=jieba.cut, stop_words=load_stopwords())
    tfidf_matrix = tfidf_model.fit_transform(docs)  
    normalized_matrix = TfidfTransformer().fit_transform(tfidf_matrix)  
    similarity = nx.from_scipy_sparse_matrix(normalized_matrix * normalized_matrix.T)  
    scores = nx.pagerank(similarity)  
    tops = sorted(scores.items(), key=lambda x: x[1], reverse=True)  
    size = min(size, len(docs))  
    indices = list(map(lambda x: x[0], tops))[:size]  
    return map(lambda idx: docs[idx], indices)  
  
  
s = '要说现在当红的90后男星，那就不得不提鹿晗、吴亦凡、杨洋、张艺兴、黄子韬、陈学冬、刘昊然，2016年他们带来不少人气爆棚的影视剧。这些90后男星不仅有颜值、有才华，还够努力，2017年他们又有哪些傲娇的作品呢？到底谁会成为2017霸屏男神，让我们拭目以待吧。鹿晗2016年参演《盗墓笔记》、《长城》、《摆渡人》等多部电影，2017年他将重心转到了电视剧。他和古力娜扎主演的古装奇幻电视剧《择天记》将在湖南卫视暑期档播出，这是鹿晗个人的首部电视剧，也是其第一次出演古装题材。该剧改编自猫腻的同名网络小说，讲述在人妖魔共存的架空世界里，陈长生(鹿晗饰演)为了逆天改命，带着一纸婚书来到神都，结识了一群志同道合的小伙伴，在国教学院打开一片新天地。吴亦凡在2017年有更多的作品推出。周星驰监制、徐克执导的春节档《西游伏魔篇》，吴亦凡扮演“有史以来最帅的”唐僧。师徒四人在取经的路上，由互相对抗到同心合力，成为无坚不摧的驱魔团队。吴亦凡还和梁朝伟、唐嫣合作动作片《欧洲攻略》，该片讲述江湖排名第一、第二的林先生(梁朝伟饰)和王小姐(唐嫣饰)亦敌亦友，二人与助手乐奇(吴亦凡饰)分别追踪盗走“上帝之手”地震飞弹的苏菲，不想却引出了欧洲黑帮、美国CIA、欧盟打击犯罪联盟特工们的全力搜捕的故事。吴亦凡2017年在电影方面有更大突破，他加盟好莱坞大片《极限特工3：终极回归》，与范·迪塞尔、甄子丹、妮娜·杜波夫等一众大明星搭档，为电影献唱主题曲《JUICE》。此外，他还参演吕克·贝松执导的科幻电影《星际特工：千星之城》，该片讲述一个发生在未来28世纪星际警察穿越时空的故事，影片有望2017年上映。'  

for i in get_abstract(s):  
    print (i) 