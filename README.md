# SPWIM
Code for the paper [Incorporating Contextual and Syntactic Structures
Improves Semantic Similarity Modeling](https://likicode.com/upload/Liu_etal_EMNLP2019.pdf) accepted at EMNLP 2019. 

We start with the [Pairwise
Word Interaction Model (PWIM)](https://www.aclweb.org/anthology/N16-1108) of He and Lin
(2016) as our base architecture. Our code is based on the implementation of PWIM at  https://github.com/castorini/castor .

### Datasets

- SICK: 
We preprocessed the original dataset and generated the ``train``, ``dev`` and ``test`` directories, each containing the following layout:
    - ``a.toks``: Each line contains one question (the same question repeats in subsequent lines for as many times as the number of candidate answers).

    - ``b.toks``: Each line contains one candidate answer.

    - ``sim.txt``: Each line contains the label (0 or 1) for the question-answer pair in the corresponding line of a.toks and b.toks respectively.

    - ``id.txt``: Each line contains the question-id for question at the corresponding line in ``a.toks``

    - ``a.parents``/ ``b.parents``: Each line contains the output of [Standford dependency parser](https://github.com/stanfordnlp/treelstm), for head words and dependency arcs separately.

    - ``a.txt`` or ``b.txt``: Each line contains original untokenized texts from the original source.
    
- Word Embeddings: [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip)

### Usage

#### Generate Tree Pickle File
```sh
$ cd vdpwi
$ python tree_loader.py --dataset_path data/sick/ --dataset sick --output_path ../
```


#### Training

```sh
$ python -m vdpwi --lr 0.0005 --optimizer rmsprop --momentum 0.05 --epochs 15  --dataset sick  --batch-size 16 --rnn-hidden-dim 256 --resultLoc result_file/ --fileID 0000 --log-interval 100 --treeFile sick_toks_tree.pkl --model_outfile sick_model
```

License
----

MIT
