from collections import namedtuple

DEBUG = True

AIML_DIR = 'aiml_parser'
AIML_FILE = 'salutations.xml'
DATA_DIR = 'data'
MODEL_DIR = 'models'
DATA_FILE = 'faq.json'
JSON = 'json'
CSV = 'csv'

# Dataframe
QUESTION = 'question'
ANSWER = 'answer'
CLEAN_QUESTION = 'clean_question'
CLEAN_ANSWER = 'clean_answer'
PROCESSED_QUESTION = 'processed_question'
PROCESSED_ANSWER = 'processed_answer'

# Trainer
QUESTION_DICTIONARY = 'question_dictionary.dict'
ANSWER_DICTIONARY = 'answer_dictionary.dict'

QUESTION_CORPUS = 'question_corpus.mm'
ANSWER_CORPUS = 'answer_corpus.mm'

QUESTION_TFIDF = 'question_tfidf.mm'
ANSWER_TFIDF = 'answer_tfidf.mm'

QUESTION_LDA_MODEL = 'question_lda_model.model'
ANSWER_LDA_MODEL = 'answer_lda_model.model'

Model = namedtuple('Model', 'column dictionary corpus tfidf model')
modes = {
    'question': Model(PROCESSED_QUESTION, QUESTION_DICTIONARY, QUESTION_CORPUS, QUESTION_TFIDF, QUESTION_LDA_MODEL),
    'answer': Model(PROCESSED_ANSWER, ANSWER_DICTIONARY, ANSWER_CORPUS, ANSWER_TFIDF, ANSWER_LDA_MODEL)
}

# BOT
UNABLE_TO_ANSWER = 'Sorry, I\'m unable to understand'
