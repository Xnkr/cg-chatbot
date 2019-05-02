import os
from operator import itemgetter

from data import preprocessing
from trainer import trainer
from webscraping import webscraper
from static.constants import *
from aiml_parser import aiml_parser


def exists_data_file():
    return os.path.exists(os.path.join(DATA_DIR, DATA_FILE))


def exists_models():
    for mode in modes:
        if not os.path.exists(os.path.join(MODEL_DIR, modes[mode].model)):
            return False
    return True


faq_df = preprocessing.get_dataframe(os.path.join(DATA_DIR, DATA_FILE), JSON)

question_dictionary = trainer.get_dictionary(QUESTION)
answer_dictionary = trainer.get_dictionary(ANSWER)

question_tfidf_model = trainer.get_tfidf(QUESTION)
answer_tfidf_model = trainer.get_tfidf(ANSWER)

question_model = trainer.get_lda_model(QUESTION)
answer_model = trainer.get_lda_model(ANSWER)

question_index = trainer.get_similarity_index(QUESTION, question_dictionary)
answer_index = trainer.get_similarity_index(ANSWER, answer_dictionary)


def domain_specific(faq_df, query):
    processed_query = trainer.preprocess(query)
    query_corpus_question = question_dictionary.doc2bow(processed_query)
    query_tfidf = question_tfidf_model[query_corpus_question]
    query_prediction = question_tfidf_model[query_corpus_question]
    sim = question_index[query_prediction]
    ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
    doc, score = ranking[0]
    if score > 0.3:
        if DEBUG:
            print(f'Your question was {faq_df.iloc[doc].question} Confidence: {score}')
        return faq_df.iloc[doc].answer, QUESTION, int(round(score * 100))
    else:
        query_corpus_answer = answer_dictionary.doc2bow(processed_query)
        query_tfidf = answer_tfidf_model[query_corpus_answer]
        query_prediction = answer_tfidf_model[query_corpus_answer]
        answer_sim = answer_index[query_prediction]
        answer_ranking = sorted(enumerate(answer_sim), key=itemgetter(1), reverse=True)
        answer_doc, answer_score = answer_ranking[0]
        if answer_score > 0.3:
            if DEBUG:
                print(f'Your question was found in answer, Confidence: {answer_score}')
            return faq_df.iloc[answer_doc].answer, ANSWER, int(round(answer_score * 100))
        else:
            return UNABLE_TO_ANSWER, None, 0


if __name__ == '__main__':
    if not exists_data_file() or not exists_models():
        if not exists_data_file():
            print('FAQ file not found. Scraping now')
            webscraper.parse("https://www.credit-suisse.com/lu/en/private-banking/services/online-banking/faq.html",
                             os.path.join(DATA_DIR, DATA_FILE))

        print('Model not found. Training now')
        trainer.train(DATA_FILE, JSON)

    print('Chatbot ready!', )
    query = 'dummy'
    while True:
        query = input('> ')
        if DEBUG and query == 'qqq':
            break
        clean_query = preprocessing.clean_text(query)
        if DEBUG:
            print(f'Cleaned query: {clean_query}')
        aiml_response = aiml_parser.get_response(clean_query)
        if aiml_response == '':
            print(domain_specific(faq_df, clean_query)[0])
        elif query.lower() == 'bye':
            print(aiml_response)
            break
        else:
            print(aiml_response)
