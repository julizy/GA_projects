
## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for analysis
import re
import nltk
import wordcloud
import contractions

## for machine learning
from sklearn import preprocessing, model_selection, feature_extraction, metrics, naive_bayes, linear_model, pipeline

## for explainer
from lime import lime_text

## for model pickle
import pickle


###############################################################################
#                               TEXT ANALYSIS                                 #
###############################################################################

def plot_distributions(dtf, x, max_cat=20, top=None, y=None, bins=None, figsize=(10,5)):
    '''
    Plot univariate and bivariate distributions.
    '''
    
    ## univariate
    if y is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(x, fontsize=15)
        ### categorical
        if dtf[x].nunique() <= max_cat:
            if top is None:
                dtf[x].reset_index().groupby(x).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            else:   
                dtf[x].reset_index().groupby(x).count().sort_values(by="index").tail(top).plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            ax.set(ylabel=None)
        ### numerical
        else:
            sns.distplot(dtf[x], hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
            ax.grid(True)
            ax.set(xlabel=None, yticklabels=[], yticks=[])

    ## bivariate
    else:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
        fig.suptitle(x, fontsize=15)
        for i in dtf[y].unique():
            sns.distplot(dtf[dtf[y]==i][x], hist=True, kde=False, bins=bins, hist_kws={"alpha":0.8}, axlabel="", ax=ax[0])
            sns.distplot(dtf[dtf[y]==i][x], hist=False, kde=True, kde_kws={"shade":True}, axlabel="", ax=ax[1])
        ax[0].set(title="histogram")
        ax[0].grid(True)
        ax[0].legend(dtf[y].unique())
        ax[1].set(title="density")
        ax[1].grid(True)
    plt.show()


def add_text_length(data, column):
    '''
    Compute different text length metrics.
    :parameter
        :param dtf: dataframe - dtf with a text column
        :param column: string - name of column containing text
    :return
        dtf: input dataframe with 2 new columns
    '''    
    
    dtf = data.copy()
    dtf['word_count'] = dtf[column].apply(lambda x: len(nltk.word_tokenize(str(x))) )
    dtf['char_count'] = dtf[column].apply(lambda x: sum(len(word) for word in nltk.word_tokenize(str(x))) )
    dtf['avg_word_length'] = dtf['char_count'] / dtf['word_count']
    print(dtf[['word_count','char_count','avg_word_length']].describe().T[["min","mean","max"]])
    return dtf


def create_stopwords(lst_langs=["english"], lst_add_words=[], lst_keep_words=[]):
    '''
    Creates a list of stopwords.
    :parameter
        :param lst_langs: list - ["english", "italian"]
        :param lst_add_words: list - list of new stopwords to add
        :param lst_keep_words: list - list words to keep (exclude from stopwords)
    :return
        stop_words: list of stop words
    '''          
    
    lst_stopwords = set()
    for lang in lst_langs:
        lst_stopwords = lst_stopwords.union( set(nltk.corpus.stopwords.words(lang)) )
    lst_stopwords = lst_stopwords.union(lst_add_words)
    lst_stopwords = list(set(lst_stopwords) - set(lst_keep_words))
    return sorted(list(set(lst_stopwords)))


def utils_preprocess_text(txt, lst_regex=None, punkt=True, lower=True, slang=True, lst_stopwords=None, stemm=False, lemm=True):
    '''
    Preprocess a string.
    :parameter
        :param txt: string - name of column containing text
        :param lst_regex: list - list of regex to remove
        :param punkt: bool - if True removes punctuations and characters
        :param lower: bool - if True convert lowercase
        :param slang: bool - if True fix slang into normal words
        :param lst_stopwords: list - list of stopwords to remove
        :param stemm: bool - whether stemming is to be applied
        :param lemm: bool - whether lemmitisation is to be applied
    :return
        cleaned text
    '''

    ## Regex (in case, before cleaning)
    if lst_regex is not None: 
        for regex in lst_regex:
            txt = re.sub(regex, '', txt)

    ## Clean 
    ### separate sentences with '. '
    txt = re.sub(r'\.(?=[^ \W\d])', '. ', str(txt))
    ### remove punctuations and characters
    txt = re.sub(r'[^\w\s]', '', txt) if punkt is True else txt
    ### strip
    txt = " ".join([word.strip() for word in txt.split()])
    ### lowercase
    txt = txt.lower() if lower is True else txt
    ### slang
    txt = contractions.fix(txt) if slang is True else txt
            
    ## Tokenize (convert from string to list)
    lst_txt = txt.split()
                
    ## Stemming (remove -ing, -ly, ...)
    if stemm is True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_txt = [ps.stem(word) for word in lst_txt]
                
    ## Lemmatization (convert the word into root word)
    if lemm is True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_txt = [lem.lemmatize(word) for word in lst_txt]

    ## Stopwords
    if lst_stopwords is not None:
        lst_txt = [word for word in lst_txt if word not in lst_stopwords]
            
    ## Back to string
    txt = " ".join(lst_txt)
    return txt


def add_preprocessed_text(data, column, lst_regex=None, punkt=False, lower=False, slang=False, lst_stopwords=None, stemm=False, lemm=False, remove_na=True):
    '''
    Adds a column of preprocessed text.
    :parameter
        :param dtf: dataframe - dtf with a text column
        :param column: string - name of column containing text
    :return
        : input dataframe with two new columns
    '''    
    
    dtf = data.copy()

    ## apply preprocess
    dtf = dtf[ pd.notnull(dtf[column]) ]
    dtf[column+"_clean"] = dtf[column].apply(lambda x: utils_preprocess_text(x, lst_regex, punkt, lower, slang, lst_stopwords, stemm, lemm))
    
    ## residuals
    dtf["check"] = dtf[column+"_clean"].apply(lambda x: len(x))
    if dtf["check"].min() == 0:
        print("--- found NAs ---")
        print(dtf[[column,column+"_clean"]][dtf["check"]==0].head())
        if remove_na is True:
            dtf = dtf[dtf["check"]>0] 
            
    return dtf.drop("check", axis=1)


def plot_wordcloud(corpus, max_words=150, max_font_size=35, figsize=(10,10)):
    '''
    Plots a wordcloud from a list of Docs or from a dictionary
    :parameter
        :param corpus: list - dtf["text"]
    '''    
    
    wc = wordcloud.WordCloud(background_color='black', max_words=max_words, max_font_size=max_font_size)
    wc = wc.generate(str(corpus)) #if type(corpus) is not dict else wc.generate_from_frequencies(corpus)     
    fig = plt.figure(num=1, figsize=figsize)
    plt.axis('off')
    plt.imshow(wc, cmap=None)
    plt.show()
    

def word_freq(corpus,vectorizer=None):
    '''
    Calculate the words frequency by vectorizer
    :parameter
        :param corpus: list - dtf["text"]
        :param vectorizer: sklearn vectorizer object, like Count or Tf-Idf
    '''    
    
    ## vectorizer
    vectorizer = feature_extraction.text.TfidfVectorizer(stop_words='english') if vectorizer is None else vectorizer
    vectorizer.fit(corpus)
    
    ## text_df
    text_vec = vectorizer.transform(corpus)
    text_df = pd.DataFrame(text_vec.todense(), columns=vectorizer.get_feature_names_out())
    
    return text_df    


def plot_word_freq(text_df,target,target_name):
    '''
    Plot the words frequency
    :parameter
        :param text_df: words frequecy df caculated by function word_freq 
        :param target: target column for classification
        :param target_name: target column name for classification
    '''    
    
    text_and_subreddit = pd.concat([text_df, target], axis=1)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

    for i, y in enumerate(target.unique()):
        df_plot = pd.DataFrame({y:text_and_subreddit[text_and_subreddit[target_name]==y][text_df.columns].sum().sort_values(ascending=False).head(15)})
        # plot
        sns.barplot(x=y, y=df_plot.index,
                    data=df_plot, orient='h', ax=axs[i], palette='viridis')
        axs[i].set(xlabel=None, ylabel=None, title=f"{y} Most Frequent Words")
    plt.show()
    
    
###############################################################################
#                         BAG OF WORDS (VECTORIZER) + ML                      #
###############################################################################

def fit_bow(corpus, vectorizer=None, vocabulary=None):
    '''
    Vectorize corpus with Bag-of-Words (classic Count or Tf-Idf variant) and plot Sparse Matrix Sample.
    :parameter
        :param corpus: list - dtf["text"]
        :param vectorizer: sklearn vectorizer object, like Count or Tf-Idf
        :param vocabulary: list of words or dict, if None it creates from scratch, else it searches the words into corpus
    :return
        sparse matrix, list of text tokenized, vectorizer, dic_vocabulary, X_names
    '''    
    
    ## vectorizer
    vectorizer = feature_extraction.text.TfidfVectorizer(max_features=None, ngram_range=(1,1), vocabulary=vocabulary) if vectorizer is None else vectorizer
    vectorizer.fit(corpus)
    
    ## sparse matrix
    print("--- creating sparse matrix ---")
    X = vectorizer.transform(corpus)
    print("shape:", X.shape)
    
    ## vocabulary
    print("--- creating vocabulary ---") if vocabulary is None else print("--- used vocabulary ---")
    dic_vocabulary = vectorizer.vocabulary_   #{word:idx for idx, word in enumerate(vectorizer.get_feature_names())}
    print(len(dic_vocabulary), "words")
    
    ## text2tokens
    print("--- tokenization ---")
    tokenizer = vectorizer.build_tokenizer()
    preprocessor = vectorizer.build_preprocessor()
    lst_text2tokens = []
    for text in corpus:
        lst_tokens = [dic_vocabulary[word] for word in tokenizer(preprocessor(text)) if word in dic_vocabulary]
        lst_text2tokens.append(lst_tokens)
    print(len(lst_text2tokens), "texts")
    
    ## plot heatmap
    fig, ax = plt.subplots(figsize=(15,5))
    sns.heatmap(X.todense()[:,np.random.randint(0,X.shape[1],100)]==0, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Sparse Matrix Sample')
    plt.show()
    return {"X":X, "lst_text2tokens":lst_text2tokens, "vectorizer":vectorizer, "dic_vocabulary":dic_vocabulary, "X_names":vectorizer.get_feature_names()}


def fit_classif(X_train, y_train, X_test, vectorizer=None, classifier=None): 
    '''
    Fits a sklearn classification model.
    :parameter
        :param X_train: feature matrix
        :param y_train: array of classes
        :param X_test: raw text
        :param vectorizer: vectorizer object - if None Tf-Idf is used
        :param classifier: model object - if None MultinomialNB is used
    :return
        fitted model and predictions
    '''    
    
    ## model pipeline
    vectorizer = feature_extraction.text.TfidfVectorizer() if vectorizer is None else vectorizer
    classifier = naive_bayes.MultinomialNB() if classifier is None else classifier
    model = pipeline.Pipeline([("vectorizer",vectorizer), ("classifier",classifier)])
    
    ## train
    if vectorizer is None:
        model.fit(X_train, y_train)
    else:
        model["classifier"].fit(X_train, y_train)
    
    ## test
    predicted = model.predict(X_test)
    predicted_prob = model.predict_proba(X_test)
    return model, predicted_prob, predicted


def explainer_lime(model, y_train, txt_instance, top=10):
    '''
    Use lime to build an a explainer.
    :parameter
        :param model: pipeline with vectorizer and classifier
        :param Y_train: array
        :param txt_instance: string - raw text
        :param top: num - top features to display
    :return
        dtf with explanations
    '''    
    
    explainer = lime_text.LimeTextExplainer(class_names=np.unique(y_train))
    explained = explainer.explain_instance(txt_instance, model.predict_proba, num_features=top) 
    explained.show_in_notebook(text=txt_instance, predict_proba=False)
    dtf_explainer = pd.DataFrame(explained.as_list(), columns=['feature','effect'])
    return dtf_explainer


###############################################################################
#                               MODEL EVALUATION                              #
###############################################################################

def evaluate_classif(y_test, predicted, predicted_prob, figsize=(15,5)):
    '''
    Evaluates a model performance.
    :parameter
        :param y_test: array
        :param predicted: array
        :param predicted_prob: array
        :param figsize: tuple - plot setting
    '''    
    
    classes = np.unique(y_test)
    y_test_array = pd.get_dummies(y_test, drop_first=False).values
    
    ## Accuracy, Precision, Recall
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob[:, 1], multi_class="ovr")
    print("Accuracy:",  round(accuracy,4))
    print("Auc:", round(auc,4))
    print("Detail:")
    print(metrics.classification_report(y_test, predicted, digits=4))
    
    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
    ax.set(xlabel="Pred", ylabel="True", xticklabels=classes, yticklabels=classes, title="Confusion matrix")
    plt.yticks(rotation=0)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ## Plot roc
    for i in range(len(classes)):
        fpr, tpr, thresholds = metrics.roc_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[0].plot(fpr, tpr, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(fpr, tpr)))
    ax[0].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax[0].set(xlim=[-0.05,1.0], ylim=[0.0,1.05], xlabel='False Positive Rate', 
              ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")
    ax[0].legend(loc="lower right")
    ax[0].grid(True)
    
    ## Plot precision-recall curve
    for i in range(len(classes)):
        precision, recall, thresholds = metrics.precision_recall_curve(y_test_array[:,i], predicted_prob[:,i])
        ax[1].plot(recall, precision, lw=3, label='{0} (area={1:0.2f})'.format(classes[i], metrics.auc(recall, precision)))
    ax[1].set(xlim=[0.0,1.05], ylim=[0.0,1.05], xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[1].legend(loc="best")
    ax[1].grid(True)
    plt.show()





