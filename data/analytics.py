from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from .preprocessing import *

if __name__ == '__main__':
    faq_df = get_dataframe('next-faq.json', type='json')
    question_tokens = ' '.join(faq_df['processed_question']).split()
    q_token_count = Counter(question_tokens)
    print(q_token_count)
    common_tokens = [token[0] for token in q_token_count.most_common(20)]
    common_count = [token[1] for token in q_token_count.most_common(20)]

    fig = plt.figure(figsize=(18, 6))
    sns.barplot(x=common_tokens, y=common_count)
    plt.title('Token count')
    plt.show()
