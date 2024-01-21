import torch
from sentence_transformers import SentenceTransformer, util

def accuracy(data, model):
    number_correct = 0
    different_questions = data.question_id.unique()
    number_of_different_questions = len(different_questions)
    k = 2
    for i in different_questions:
        temp_question_2 = data.loc[data['question_id'] == i]
        temp_question_1_vector = model.encode(temp_question_2.question_1.iloc[0])
        temp_question_2_vector = model.encode(list(temp_question_2.question_2))

        document_scores = util.cos_sim(temp_question_1_vector,
                                       temp_question_2_vector)[0]

        top_results = torch.topk(document_scores,k=2)

        top_results = top_results[1].tolist()[0]
        if temp_question_2.iloc[top_results].label == 1:
            number_correct += 1
    return number_correct, number_of_different_questions