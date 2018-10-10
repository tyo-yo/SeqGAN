# SeqGAN with keras

## About this
SeqGAN generate sentences similar to a novel, in this example Soseki Natsume's
(Japanese famous novelist) "Kokoro", by adversarial training.

Look at the [paper](https://arxiv.org/abs/1609.05473) for detail.

In this example, disciminator model differs from paper because keras don't support masking in Conv1D, so disciminator uses LSTM here.

## Usage
[main.ipynb](https://github.com/tyo-yo/SeqGAN/blob/master/main.ipynb) shows usage and result.

## Requirements
- Python 3.6.6
- Keras==2.2.2
- tensorflow==1.10.0


## Test
> python setup.py test
