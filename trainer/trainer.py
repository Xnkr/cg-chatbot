import spacy
import os
from gensim.models import TfidfModel
from gensim import corpora, similarities
from gensim.models import LdaModel, LdaMulticore, LsiModel, HdpModel
from data.preprocessing import *

nlp = spacy.load('en_core_web_sm', disable=['parser', 'tagger'])


def preprocess(text):
    document = nlp(text)
    lemmas = [token.lemma_ for token in document if not token.is_stop]
    return lemmas


def train():
    faq_df = get_dataframe(os.path.join('data', 'next-faq.json'), type='json')
    faq_df = clean_data(faq_df)
    faq_df['processed_question'] = faq_df['clean_question'].apply(preprocess)
    faq_df['processed_answer'] = faq_df['clean_answer'].apply(preprocess)
    print('Preprocessing Done')

    question_dictionary = corpora.Dictionary(faq_df['processed_question'])
    question_dictionary.save(os.path.join('models', 'question_dictionary.dict'))
    question_corpus = faq_df["processed_question"].map(question_dictionary.doc2bow)
    corpora.MmCorpus.serialize(os.path.join('models', 'question_corpus.mm'), question_corpus)
    question_tfidf_model = TfidfModel(question_corpus)
    question_tfidf_model.save(os.path.join('models', 'question_tfidf.mm'))
    question_tfidf = question_tfidf_model[question_corpus]

    answer_dictionary = corpora.Dictionary(faq_df['processed_answer'])
    answer_dictionary.save(os.path.join('models', 'answer_dictionary.dict'))
    answer_corpus = faq_df['processed_answer'].map(answer_dictionary.doc2bow)
    corpora.MmCorpus.serialize(os.path.join('models', 'answer_corpus.mm'), answer_corpus)
    answer_tfidf_model = TfidfModel(answer_corpus)
    answer_tfidf_model.save(os.path.join('models', 'answer_tfidf.mm'))
    answer_tfidf = answer_tfidf_model[answer_corpus]
    print('TF-IDF generated')

    # lsi_model = LsiModel(corpus=question_tfidf, id2word=q_dictionary, num_topics=7, decay=0.5)
    # lsi_model.save('lsi_model.model')

    # hdp_model = HdpModel(quesiton_tfidf, q_dictionary)
    # hdp_model.save('hdp_model.model')
    question_lda_model = LdaMulticore(corpus=question_tfidf, id2word=question_dictionary, random_state=100,
                                      num_topics=7,
                                      passes=10, chunksize=1000, batch=False, alpha='asymmetric', decay=0.5, offset=64,
                                      eta=None, eval_every=0, iterations=100, gamma_threshold=0.001)
    question_lda_model.save(os.path.join('models', 'question_lda_model.model'))

    answer_lda_model = LdaMulticore(corpus=answer_tfidf, id2word=answer_dictionary, random_state=100, num_topics=7,
                                    passes=10, chunksize=1000, batch=False, alpha='asymmetric', decay=0.5, offset=64,
                                    eta=None, eval_every=0, iterations=100, gamma_threshold=0.001)
    answer_lda_model.save(os.path.join('models', 'answer_lda_model.model'))
    print('LDA Models generated')


def get_dictionary(mode='question'):
    return corpora.Dictionary.load(os.path.join('models', mode + '_dictionary.dict'))


def get_corpus(mode='question'):
    return corpora.MmCorpus(os.path.join('models', mode + '_corpus.mm'))


def get_tfidf(mode='question'):
    return TfidfModel.load(os.path.join('models', mode + '_tfidf.mm'))


def get_lda_model(model='question'):
    return LdaMulticore.load(os.path.join('models', model + '_lda_model.model'))


def get_similarity_index(mode, dictionary):
    question_corpus = get_corpus(mode)
    return similarities.MatrixSimilarity(question_corpus, num_features=len(dictionary))
