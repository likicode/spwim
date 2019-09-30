import torch
import torch.nn.functional as F

from .evaluator import Evaluator


class QuoraEvaluator(Evaluator):

    def get_scores(self):
        self.model.eval()
        test_cross_entropy_loss_vdpwi = 0
        test_cross_entropy_loss_tree = 0
        acc_total = 0

        with torch.no_grad():
            for batch in self.data_loader:
                sent1, sent2 = self.get_sentence_embeddings(batch)

                vdpwi_output, tree_output = self.model(sent1, sent2, batch.ext_feats, batch.dataset.word_to_doc_cnt, batch.sentence_1_raw, batch.sentence_2_raw)
                test_cross_entropy_loss_vdpwi += F.cross_entropy(vdpwi_output, batch.label, size_average=False).item()
                test_cross_entropy_loss_tree += F.cross_entropy(tree_output, batch.label, size_average=False).item()

                true_label = batch.label.detach().cpu().numpy()
                prediction = torch.max((vdpwi_output.detach().exp() + tree_output.detach().exp()) / 2, 1)[1]

                acc_total += ((true_label == prediction)).sum().item()

                del vdpwi_output, tree_output

            test_cross_entropy_loss = test_cross_entropy_loss_vdpwi + test_cross_entropy_loss_tree
            test_cross_entropy_loss /= len(batch.dataset.examples)

            accuracy = acc_total / len(self.data_loader.dataset.examples)

        return [accuracy, test_cross_entropy_loss], ['accuracy', 'cross entropy loss']

