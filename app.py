import os
from trainer import trainer
from data import preprocessing
from operator import itemgetter

if __name__ == '__main__':
    if not os.path.exists(os.path.join('models', 'question_lda_model.model')) or not os.path.exists(
            os.path.join('models', 'answer_lda_model.model')):
        print('Model not found. Training now')
        trainer.train()

    faq_df = preprocessing.get_dataframe(os.path.join('data', 'next-faq.json'), type='json')

    question_dictionary = trainer.get_dictionary('question')
    answer_dictionary = trainer.get_dictionary('answer')

    question_tfidf_model = trainer.get_tfidf('question')
    answer_tfidf_model = trainer.get_tfidf('answer')

    question_model = trainer.get_lda_model('question')
    answer_model = trainer.get_lda_model('answer')

    question_index = trainer.get_similarity_index('question', question_dictionary)
    answer_index = trainer.get_similarity_index('answer', answer_dictionary)

    print('Chatbot ready!', )
    query = 'dummy'
    while True:
        query = input('> ')
        if query == 'quit':
            print('bye')
            break
        clean_query = preprocessing.context_specific_cleaning(query)
        processed_query = trainer.preprocess(clean_query)
        query_corpus_question = question_dictionary.doc2bow(processed_query)
        query_tfidf = question_tfidf_model[query_corpus_question]
        sim = question_index[query_tfidf]
        ranking = sorted(enumerate(sim), key=itemgetter(1), reverse=True)
        doc, score = ranking[0]
        if score > 0.3:
            print(f'Your question was {faq_df.iloc[doc].question} Confidence: {score}')
            print(faq_df.iloc[doc].answer,)
        else:
            query_corpus_answer = answer_dictionary.doc2bow(processed_query)
            query_tfidf = answer_tfidf_model[query_corpus_answer]
            answer_sim = answer_index[query_tfidf]
            answer_ranking = sorted(enumerate(answer_sim), key=itemgetter(1), reverse=True)
            answer_doc, answer_score = answer_ranking[0]
            print(f'Your question was found in answer Confidence: {answer_score}')
            if answer_score > 0.3:
                if answer_score * 0.3 > score * 0.7:
                    print(faq_df.iloc[answer_doc].answer,)
                else:
                    print(faq_df.iloc[doc].answer,)
            else:
                print('Sorry, I\'m unable to understand')

