import torch
import torch.nn.functional as F

from .evaluator import Evaluator
from sklearn.metrics import f1_score

def URL_maxF1_eval(predict_result, test_data_label):
    test_data_label = [item >= 1 for item in test_data_label]
    counter = 0
    tp = 0.0
    fp = 0.0
    fn = 0.0
    tn = 0.0

    for i, t in enumerate(predict_result):

        if t > 0.5:
            guess = True
        else:
            guess = False
        label = test_data_label[i]
        # print guess, label
        if guess == True and label == False:
            fp += 1.0
        elif guess == False and label == True:
            fn += 1.0
        elif guess == True and label == True:
            tp += 1.0
        elif guess == False and label == False:
            tn += 1.0
        if label == guess:
            counter += 1.0

    try:
        P = tp / (tp + fp)
        R = tp / (tp + fn)
        F = 2 * P * R / (P + R)
    except:
        P = 0
        R = 0
        F = 0

    accuracy = counter / len(predict_result)

    maxF1 = 0
    P_maxF1 = 0
    R_maxF1 = 0
    probs = predict_result
    sortedindex = sorted(range(len(probs)), key=probs.__getitem__)
    sortedindex.reverse()

    truepos = 0
    falsepos = 0
    for sortedi in sortedindex:
        if test_data_label[sortedi] == True:
            truepos += 1
        elif test_data_label[sortedi] == False:
            falsepos += 1
        precision = 0
        if truepos + falsepos > 0:
            precision = truepos / (truepos + falsepos)

        if (tp + fn) > 0:
            recall = truepos / (tp + fn)
        else:
            recall = 0
        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
            if f1 > maxF1:
                # print probs[sortedi]
                maxF1 = f1
                P_maxF1 = precision
                R_maxF1 = recall
    # print("PRECISION: {}, RECALL: {}, max_F1: {}".format(P_maxF1, R_maxF1, maxF1))
    return maxF1

class TwitterEvaluator(Evaluator):

    def get_scores(self):
        self.model.eval()
        test_cross_entropy_loss = 0
        acc_total = 0
        true_labels = []
        predictions = []
        test_cross_entropy_loss_vdpwi = 0
        test_cross_entropy_loss_tree = 0

        with torch.no_grad():
            for batch in self.data_loader:
                sent1, sent2 = self.get_sentence_embeddings(batch)

                vdpwi_output, tree_output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
                test_cross_entropy_loss_vdpwi += F.cross_entropy(vdpwi_output, batch.label, size_average=False).item()
                test_cross_entropy_loss_tree += F.cross_entropy(tree_output, batch.label, size_average=False).item()

                true_label = batch.label.detach().cpu().numpy()
                prediction = torch.max((vdpwi_output.detach().exp() + tree_output.detach().exp())/2, 1)[1]

                true_labels.extend(true_label)
                predictions.extend(prediction)

                del vdpwi_output, tree_output

            test_cross_entropy_loss /= len(batch.dataset.examples)

            f1 = URL_maxF1_eval(predictions, true_labels)

        return [f1, test_cross_entropy_loss], ['F1', 'cross entropy loss']

