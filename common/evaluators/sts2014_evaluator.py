from scipy.stats import pearsonr, spearmanr
import torch
import torch.nn.functional as F

from .evaluator import Evaluator

def calculate_r(predictions, labels, idx_list):
    pearson_value = []
    spear_value = []
    val_len = []
    for idx in idx_list:
        prediction = [predictions[i] for i in idx]
        label = [labels[i] for i in idx]
        pearson = pearsonr(prediction, label)[0]
        spear = spearmanr(prediction, label)[0]
        pearson_value.append(pearson)
        spear_value.append(spear)
        val_len.append(len(idx))
    sum_len = sum(val_len) * 1.0
    sum_pear = 0
    sum_spear = 0
    for _pear, _spear, weight in zip(pearson_value, spear_value, val_len):
        sum_pear += _pear * weight
        sum_spear += _spear * weight
        print(_pear)
    return (sum_pear/sum_len, sum_spear/sum_len)

class STS2014Evaluator(Evaluator):

    def get_scores(self):
        self.model.eval()
        num_classes = self.dataset_cls.NUM_CLASSES
        test_kl_div_loss = 0
        predictions = []
        true_labels = []
        test_loss_vdpwi = 0
        test_loss_tree = 0

        with torch.no_grad():
            for batch in self.data_loader:
                sent1, sent2 = self.get_sentence_embeddings(batch)
                vdpwi_output, tree_output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
                test_loss_vdpwi += F.kl_div(vdpwi_output, batch.label, size_average=False).item()
                test_loss_tree += F.kl_div(tree_output, batch.label, size_average=False).item()
                test_kl_div_loss += test_loss_vdpwi + test_loss_tree

                predict_classes = batch.label.new_tensor(torch.arange(0, num_classes)).expand(self.batch_size, num_classes)
                # handle last batch which might have smaller size
                if len(predict_classes) != len(batch.sentence_1):
                    predict_classes = batch.label.new_tensor(torch.arange(0, num_classes)).expand(len(batch.sentence_1), num_classes)

                true_labels.append((predict_classes * batch.label.detach()).sum(dim=1))
                predictions.append((predict_classes * (vdpwi_output.detach().exp() + tree_output.detach().exp())/2).sum(dim=1))

                del vdpwi_output
                del tree_output

            predictions = torch.cat(predictions)
            true_labels = torch.cat(true_labels)
            mse = F.mse_loss(predictions, true_labels).item()
            test_kl_div_loss /= len(batch.dataset.examples)
            predictions = predictions.cpu().numpy()
            true_labels = true_labels.cpu().numpy()
            pearson_r = pearsonr(predictions, true_labels)[0]
            spearman_r = spearmanr(predictions, true_labels)[0]
            print("train pearson: {}".format(pearson_r))
        return [pearson_r, spearman_r, mse, test_kl_div_loss], ['pearson_r', 'spearman_r', 'mse', 'KL-divergence loss']

    def get_final_prediction_and_label(self, batch_predictions, batch_labels):
        num_classes = self.dataset_cls.NUM_CLASSES
        predict_classes = batch_labels.new_tensor(torch.arange(1, num_classes + 1)).expand(batch_predictions.size(0), num_classes)

        predictions = (predict_classes * batch_predictions.exp()).sum(dim=1)
        true_labels = (predict_classes * batch_labels).sum(dim=1)

        return predictions, true_labels

    def get_sts_test_scores(self):
        self.model.eval()
        num_classes = self.dataset_cls.NUM_CLASSES
        test_kl_div_loss = 0
        predictions = []
        true_labels = []
        test_loss_vdpwi = 0
        test_loss_tree = 0

        #type category: deft-forum','deft-news','headlines','images','OnWN','tweet-news'
        type_list = [[] for i in range(6)]

        with torch.no_grad():
            for batch in self.data_loader:
                # Select embedding
                pid = batch.id.detach().cpu().numpy()
                batch_size = len(pid)
                for k in range(batch_size):
                    type_id = pid[k]
                    type_list[type_id].append(k + batch_size * i)

                sent1, sent2 = self.get_sentence_embeddings(batch)

                vdpwi_output, tree_output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
                test_loss_vdpwi += F.kl_div(vdpwi_output, batch.label, size_average=False).item()
                test_loss_tree += F.kl_div(tree_output, batch.label, size_average=False).item()
                test_kl_div_loss += test_loss_vdpwi + test_loss_tree

                predict_classes = batch.label.new_tensor(torch.arange(0, num_classes)).expand(self.batch_size, num_classes)
                # handle last batch which might have smaller size
                if len(predict_classes) != len(batch.sentence_1):
                    predict_classes = batch.label.new_tensor(torch.arange(0, num_classes)).expand(len(batch.sentence_1), num_classes)

                true_labels.append((predict_classes * batch.label.detach()).sum(dim=1))
                predictions.append((predict_classes * output.detach().exp()).sum(dim=1))
                del output

            predictions = torch.cat(predictions)
            true_labels = torch.cat(true_labels)
            mse = F.mse_loss(predictions, true_labels).item()

            test_kl_div_loss /= len(batch.dataset.examples)
            predictions = predictions.cpu().numpy()
            true_labels = true_labels.cpu().numpy()

            pearson_r, spearman_r = calculate_r(predictions, true_labels, type_list)
            print("test pearson: {}".format(pearson_r))
        return [pearson_r, spearman_r, mse, test_kl_div_loss], ['pearson_r', 'spearman_r', 'mse', 'KL-divergence']
