from .context import unittest, os, GeneratorPretrainingGenerator

top = os.getcwd()

class TestGeneratorPretrainingGenerator(unittest.TestCase):
    def sub_test(self, actual, expected, msg=None):
        with self.subTest(actual=actual, expected=expected):
            self.assertEqual(actual, expected, msg=msg)

    def test_generator_pretraining_generator(self):
        gen = GeneratorPretrainingGenerator(
            os.path.join(top, 'data', 'kokoro_parsed.txt'),
            B=8,
            shuffle=False)
        gen.reset()
        x, y_true = gen.next()

        expected_text = ['<S>', '私', 'は', 'その', '人', 'を', '常に', '先生', 'と', '呼ん', 'で', 'い', 'た', '。', '</S>']
        length = len(expected_text)
        actual_text = [gen.id2word[id] for id in x[0][:length]]
        self.sub_test(actual_text, expected_text, msg='x text test')

        expected_text = ['<S>','だから', 'ここ', 'でも', 'ただ', '先生', 'と', '書く', 'だけ', 'で', '本名', 'は', '打ち明け', 'ない', '。', '</S>']
        expected_ids = [gen.word2id[word] for word in expected_text]
        actual_ids = x[1][:len(expected_ids)]
        result = (actual_ids == expected_ids).all()
        self.assertTrue(result, msg='x ids test')

        self.sub_test(gen.len, 4267//8)
