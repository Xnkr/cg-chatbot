import os

import spacy
from gensim import corpora, similarities
from gensim.models import LdaMulticore
from gensim.models import TfidfModel

from data.preprocessing import *
from static.constants import *

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger'])


def preprocess(text):
    document = nlp(text)
    lemmas = [token.lemma_ for token in document if not token.is_stop]
    return lemmas


def delete_previous_models():
    for file in os.listdir(MODEL_DIR):
        if DEBUG:
            print(f'Removing file {file}')
        if os.path.isfile(os.path.join(MODEL_DIR, file)):
            os.remove(os.path.join(MODEL_DIR, file))


def train(file=DATA_FILE, type=JSON):
    delete_previous_models()

    faq_df = get_dataframe(os.path.join(DATA_DIR, file), type=type)
    faq_df = clean_data(faq_df)
    faq_df[PROCESSED_QUESTION] = faq_df[CLEAN_QUESTION].apply(preprocess)
    faq_df[PROCESSED_ANSWER] = faq_df[CLEAN_ANSWER].apply(preprocess)
    print('Preprocessing Done')
    if DEBUG:
        print(faq_df.head())

    for mode in modes:
        model = modes[mode]
        dictionary = corpora.Dictionary(faq_df[model.column])
        dictionary.save(os.path.join(MODEL_DIR, model.dictionary))
        corpus = faq_df[model.column].map(dictionary.doc2bow)
        if DEBUG:
            print(f'{model.corpus} generated')
            print(corpus.head())
        corpora.MmCorpus.serialize(os.path.join(MODEL_DIR, model.corpus), corpus)
        tfidf_model = TfidfModel(corpus)
        if DEBUG:
            print(f'{model.tfidf} generated')
        tfidf_model.save(os.path.join(MODEL_DIR, model.tfidf))
        tfidf = tfidf_model[corpus]
        lda_model = LdaMulticore(corpus=tfidf, id2word=dictionary, random_state=100,
                                 num_topics=7,
                                 passes=10, chunksize=1000, batch=False, alpha='asymmetric', decay=0.5,
                                 offset=64,
                                 eta=None, eval_every=0, iterations=100, gamma_threshold=0.001)
        lda_model.save(os.path.join(MODEL_DIR, model.model))
        if DEBUG:
            print(f'{model.model} generated')
            print(lda_model.print_topics(5))
    print('Training completed')
    # lsi_model = LsiModel(corpus=question_tfidf, id2word=q_dictionary, num_topics=7, decay=0.5)
    # lsi_model.save('lsi_model.model')

    # hdp_model = HdpModel(quesiton_tfidf, q_dictionary)
    # hdp_model.save('hdp_model.model')


def get_dictionary(mode=QUESTION):
    return corpora.Dictionary.load(os.path.join(MODEL_DIR, modes[mode].dictionary))


def get_corpus(mode=QUESTION):
    return corpora.MmCorpus(os.path.join(MODEL_DIR, modes[mode].corpus))


def get_tfidf(mode=QUESTION):
    return TfidfModel.load(os.path.join(MODEL_DIR, modes[mode].tfidf))


def get_lda_model(mode=QUESTION):
    return LdaMulticore.load(os.path.join(MODEL_DIR, modes[mode].model))


def get_similarity_index(mode, dictionary):
    corpus = get_corpus(mode)
    return similarities.MatrixSimilarity(corpus, num_features=len(dictionary))
