from .evaluators.sick_evaluator import SICKEvaluator
from .evaluators.trecqa_evaluator import TRECQAEvaluator
from .evaluators.wikiqa_evaluator import WikiQAEvaluator
from .evaluators.pit2015_evaluator import PIT2015Evaluator
from .evaluators.snli_evaluator import SNLIEvaluator
from .evaluators.sts2014_evaluator import STS2014Evaluator
from .evaluators.quora_evaluator import QuoraEvaluator
from .evaluators.twitter_evaluator import TwitterEvaluator
from nce.nce_pairwise_mp.evaluators.trecqa_evaluator import TRECQAEvaluatorNCE
from nce.nce_pairwise_mp.evaluators.wikiqa_evaluator import WikiQAEvaluatorNCE



class EvaluatorFactory(object):
    """
    Get the corresponding Evaluator class for a particular dataset.
    """
    evaluator_map = {
        'sick': SICKEvaluator,
        'trecqa': TRECQAEvaluator,
        'wikiqa': WikiQAEvaluator,
        'pit2015': PIT2015Evaluator,
        'twitterurl': PIT2015Evaluator,
        'SNLI': SNLIEvaluator,
        'sts2014': STS2014Evaluator,
        'Quora': QuoraEvaluator,
        'Twitter': TwitterEvaluator
    }

    evaluator_map_nce = {
        'trecqa': TRECQAEvaluatorNCE,
        'wikiqa': WikiQAEvaluatorNCE
    }

    @staticmethod
    def get_evaluator(dataset_cls, model, embedding, data_loader, batch_size, device, nce=False, keep_results=False):
        if data_loader is None:
            return None

        if nce:
            evaluator_map = EvaluatorFactory.evaluator_map_nce
        else:
            evaluator_map = EvaluatorFactory.evaluator_map

        if not hasattr(dataset_cls, 'NAME'):
            raise ValueError('Invalid dataset. Dataset should have NAME attribute.')

        if dataset_cls.NAME not in evaluator_map:
            raise ValueError('{} is not implemented.'.format(dataset_cls))

        return evaluator_map[dataset_cls.NAME](
            dataset_cls, model, embedding, data_loader, batch_size, device, keep_results
        )
