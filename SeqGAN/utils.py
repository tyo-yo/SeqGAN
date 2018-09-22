import numpy as np
import random
import linecache
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical

class Vocab:
    def __init__(self, word2id, unk_token):
        self.word2id = dict(word2id)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.unk_token = unk_token

    def build_vocab(self, sentences, min_count=1):
        word_counter = {}
        for sentence in sentences:
            for word in sentence:
                word_counter[word] = word_counter.get(word, 0) + 1

        for word, count in sorted(word_counter.items(), key=lambda x: -x[1]):
            if count < min_count:
                break
            _id = len(self.word2id)
            self.word2id.setdefault(word, _id)
            self.id2word[_id] = word

        self.raw_vocab = {w: word_counter[w] for w in self.word2id.keys() if w in word_counter}

    def sentence_to_ids(self, sentence):
        return [self.word2id[word] if word in self.word2id else self.word2id[self.unk_token] for word in sentence]

def load_data(file_path):
    '''
    テキストファイルからデータを読み込む関数
    # Arguments:
        file_path: str
    # Returns:
        data: list of list of str, data[i] means a sentence, data[i][j] means a
            word.
    '''
    data = []
    for line in open(file_path, encoding='utf-8'):
        words = line.strip().split()  # スペースで単語を分割
        data.append(words)
    return data

def sentence_to_ids(vocab, sentence, UNK=3):
    '''
    単語(str)のリストをID(int)のリストに変換する関数
    # Arguments:
        vocab: SeqGAN.utils.Vocab
        sentence: list of str
    # Returns:
        ids: list of int
    '''
    ids = [vocab.word2id.get(word, UNK) for word in sentence]
    # ids += [EOS]  # EOSを加える
    return ids

def pad_seq(seq, max_length, PAD=0):
    """
    Paddingを行う関数
    :param seq: list of int, 単語のインデックスのリスト
    :param max_length: int, バッチ内の系列の最大長
    :return seq: list of int, 単語のインデックスのリスト
    """
    seq += [PAD for i in range(max_length - len(seq))]
    return seq

def print_ids(ids, vocab, verbose=True, exclude_mark=True, PAD=0, BOS=1, EOS=2):
    '''
    :param ids: list of int, idのリスト
    :param vocab: idとwordを紐付けるVocab
    :param verbose(optional): Trueの場合sentenceをprintする。
    :return sentence: list of str
    '''
    sentence = []
    for i, id in enumerate(ids):
        word = vocab.id2word[id]
        if exclude_mark and id == EOS:
            break
        if exclude_mark and id in (BOS, PAD):
            continue
        sentence.append(sentence)
    if verbose:
        print(sentence)
    return sentence


class GeneratorPretrainingGenerator(Sequence):
    '''
    Generate generator pretraining data.
    # Arguments
        path: str, path to data x
        B: int, batch size
        T (optional): int or None, default is None.
            if int, T is the max length of sequential data.
        min_count (optional): int, minimum of word frequency for building vocabrary
        shuffle (optional): bool

    # Params
        PAD, BOS, EOS, UNK: int, id
        PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN: str
        B, min_count: int
        vocab: Vocab
        word2id: Vocab.word2id
        id2word: Vocab.id2word
        raw_vocab: Vocab.raw_vocab
        V: the size of vocab
        n_data: the number of rows of data

    # Examples
        generator = VAESequenceGenerator('./data/train_x.txt', 32)
        x, y_true = generator.__getitem__(idx=11)
        print(x[0])
        >>> 8, 10, 6, 3, 2, 0, 0, ..., 0
        print(y_true[0])
        >>> 1, 8, 10, 6, 3, 2, 0, ..., 0

        id2word = generator.id2word

        x_words = [id2word[id] for id in x[0]]
        print(x_words)
        >>> <S> I have a <UNK> </S> <PAD> ... <PAD>

        y_true_words = [id2word[id] for id in y_true[0]]
        print(y_true_words)
        >>> I have a <UNK> </S> <PAD> <PAD> ... <PAD>
    '''
    def __init__(self, path, B, min_count=1, shuffle=True):
        self.PAD = 0
        self.BOS = 1
        self.EOS = 2
        self.UNK = 3
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<S>'
        self.EOS_TOKEN = '</S>'
        self.path = path
        self.B = B
        self.min_count = min_count

        default_dict = {
            self.PAD_TOKEN: self.PAD,
            self.BOS_TOKEN: self.BOS,
            self.EOS_TOKEN: self.EOS,
            self.UNK_TOKEN: self.UNK,
        }
        self.vocab = Vocab(default_dict, self.UNK_TOKEN)
        sentences = load_data(path)
        self.vocab.build_vocab(sentences, self.min_count)

        self.word2id = self.vocab.word2id
        self.id2word = self.vocab.id2word
        self.raw_vocab = self.vocab.raw_vocab
        self.V = len(self.vocab.word2id)
        with open(path, 'r', encoding='utf-8') as f:
            self.n_data = sum(1 for line in f)            
        self.shuffle = shuffle
        self.idx = 0
        self.len = self.__len__()
        self.reset()


    def __len__(self):
        return self.n_data // self.B

    def __getitem__(self, idx):
        '''
        Get generator pretraining data batch.
        # Arguments:
            idx: int, index of batch
        # Returns:
            None: no input is needed for generator pretraining.
            x: numpy.array, shape = (B, max_length)
            y_true: numpy.array, shape = (B, max_length)
                max_length is the max length of sequence in the batch.
                if length smaller than max_length, the data will be padded.
        '''
        x, y_true = [], []
        start = idx * self.B + 1
        end = (idx + 1) * self.B + 1
        max_length = 0
        for i in range(start, end):
            if self.shuffle:
                idx = self.shuffled_indices[i]
            else:
                idx = i
            sentence = linecache.getline(self.path, idx) # str
            words = sentence.strip().split()  # list of str
            ids = sentence_to_ids(self.vocab, words) # list of ids

            ids_x, ids_y_true = [], []

            ids_x.append(self.BOS)
            ids_x.extend(ids)
            ids_x.append(self.EOS) # ex. [BOS, 8, 10, 6, 3, EOS]
            x.append(ids_x)

            ids_y_true.extend(ids)
            ids_y_true.append(self.EOS) # ex. [8, 10, 6, 3, EOS]
            y_true.append(ids_y_true)

            max_length = max(max_length, len(ids_x))

        x = [pad_seq(sen, max_length) for sen in x]
        x = np.array(x, dtype=np.int32)

        y_true = [pad_seq(sen, max_length) for sen in y_true]
        y_true = np.array(y_true, dtype=np.int32)

        return (x, y_true)

    def __iter__(self):
        return self

    def next(self):
        if self.idx >= self.len:
            self.reset()
            raise StopIteration
        x, y_true = self.__getitem__(self.idx)
        self.idx += 1
        return x, y_true

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.shuffled_indices = np.arange(self.n_data)
            random.shuffle(self.shuffled_indices)

    def on_epoch_end(self):
        self.reset()
        pass
