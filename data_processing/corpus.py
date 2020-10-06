import os
import json
import pickle
import numpy as np
from collections import Counter
from gensim.models.wrappers import FastText


class KGCVAECorpus:
    bod_utt = ['<s>', '<d>', '</s>']
    reserved_token_for_dialog = ['<s>', '<d>', '</s>']
    reserved_token_for_gen = ['<pad>', '<unk>', '<sos>', '<eos>']
    word2vec = []

    def __init__(self, config):
        # config 파일에서 가져온 값과 가공한 값을 구분하기 위한 내부 클래스
        class Config:
            def __init__(self):
                dir_path = config['data_dir']

                train_file_name = config['train_filename']
                self.train_file = os.path.join(dir_path, train_file_name)

                test_file_name = config['test_filename']
                self.test_file = os.path.join(dir_path, test_file_name)

                valid_file_name = config['valid_filename']
                self.valid_file = os.path.join(dir_path, valid_file_name)

                self.word2vec_path = config['word2vec_path']
                self.word2vec_dim = config['embed_size']

                self.is_load_vocab = config.get('load_vocab', False)
                self.vocab_path = config['vocab_path']
                self.max_vocab_count = config['max_vocab_count']

        self.cf = Config()

        train_corpus_data = self._get_corpus_data(self.cf.train_file)
        test_corpus_data = self._get_corpus_data(self.cf.test_file)
        valid_corpus_data = self._get_corpus_data(self.cf.valid_file)

        # CorpusPack Object
        print('Start process train corpus.')
        self.train_corpus = self._process(train_corpus_data)

        print('Start process test corpus.')
        self.test_corpus = self._process(test_corpus_data)

        print('Start process valid corpus.')
        self.valid_corpus = self._process(valid_corpus_data)

        # Vocab info object
        class Vocab:
            vocab = None
            rev_vocab = None
            unk_id = None
            topic_vocab = None
            rev_topic_vocab = None
            senti_vocab = None
            rev_senti_vocab = None

        self.vc = Vocab()

        if self.cf.is_load_vocab:
            print('Start loading vocab.')
            self._load_vocab()
        else:
            print('Start building vocab.')
            self._build_vocab()
            self._save_vocab()

        self._load_word2vec()
        print('Done loading corpus.')

    @staticmethod
    def _get_corpus_data(file):
        with open(file, 'r') as reader:
            if file.endswith('.json'):
                return json.load(reader)
            elif file.endswith('.jsonl'):
                jsonl_content = reader.read().strip()
                return [json.loads(jline) for jline in jsonl_content.split('\n')]
            else:
                raise ValueError('Not supported file format for train data. Please check corpus config settings.')

    def _process(self, corpus_data):
        # Corpus info object
        class Corpus:
            dialog = []
            meta = []
            utts = []

        cp = Corpus()

        utt_lengths = []

        for session_data in corpus_data:
            session_utts = session_data['utts']
            a_info = session_data['A']
            b_info = session_data['B']

            lower_utts = [(caller, utt, senti) for caller, utt, raw_sentence, _, senti in session_utts]
            utt_lengths.extend([len(utt) for caller, utt, senti in lower_utts])

            def get_vec_meta(caller_info):
                # 연령대 별로 0, 1, 2 값을 가짐
                vec_age_meta = [0, 0, 0]
                vec_age_meta[caller_info['age']] = 1
                vec_sex_meta = [0, 0]
                vec_sex_meta[caller_info['sex']] = 1
                return vec_age_meta + vec_sex_meta

            topic = session_data['topic'] + '_' + a_info['relation_group']
            meta = (get_vec_meta(a_info), get_vec_meta(b_info), topic)

            # dialog = (utt, caller, senti) List
            dialog = [(self.bod_utt, 0, None)] + [(utt, int(caller == 'B'), senti) for caller, utt, senti in lower_utts]
            cp.utts.extend([self.bod_utt] + [utt for caller, utt, senti in lower_utts])
            cp.dialog.append(dialog)
            cp.meta.append(meta)

        print('Max utt len %d, mean utt len %.2f' % (np.max(utt_lengths), float(np.mean(utt_lengths))))
        return cp

    def _load_vocab(self):
        with open(self.cf.vocab_path, 'rb') as reader:
            vocab = pickle.load(reader)
        self.vc.vocab = vocab['vocab']
        self.vc.rev_vocab = vocab['rev_vocab']
        self.vc.unk_id = self.vc.rev_vocab['<unk>']
        self.vc.topic_vocab = vocab['topic_vocab']
        self.vc.rev_topic_vocab = vocab['rev_topic_vocab']
        self.vc.senti_vocab = vocab['senti_vocab']
        self.vc.rev_senti_vocab = vocab['rev_senti_vocab']

    def _build_vocab(self):
        all_words = []
        for tokens in self.train_corpus.utts:
            all_words.extend(tokens)
        vocab_count = Counter(all_words).most_common()
        raw_vocab_size = len(vocab_count)
        discard_wc = np.sum([count for token, count in vocab_count[self.cf.max_vocab_count:]])
        vocab_count = vocab_count[:self.cf.max_vocab_count]
        oov_rate = float(discard_wc) / len(all_words)

        # Create vocabulary list sorted by count
        self.vc.vocab = self.reserved_token_for_gen + [token for token, count in vocab_count]
        self.vc.rev_vocab = {token: idx for idx, token in enumerate(self.vc.vocab)}
        self.vc.unk_id = self.vc.rev_vocab['<unk>']

        # Create topic vocab
        all_topics = [topic for caller_a, caller_b, topic in self.train_corpus.meta]
        self.vc.topic_vocab = [token for token, count in Counter(all_topics).most_common()]
        self.vc.rev_topic_vocab = {token: idx for idx, token in enumerate(self.vc.topic_vocab)}

        # Create senti vocab
        all_sentiments = []
        for dialog in self.train_corpus.dialog:
            all_sentiments.extend([senti for caller, utt, senti in dialog if senti is not None])
        self.vc.senti_vocab = [token for token, count in Counter(all_sentiments).most_common()]
        self.vc.rev_senti_vocab = {token: idx for idx, token in enumerate(self.vc.senti_vocab)}

    def _save_vocab(self):
        vocab = {
            'vocab': self.vc.vocab,
            'rev_vocab': self.vc.rev_vocab,
            'topic_vocab': self.vc.topic_vocab,
            'rev_topic_vocab': self.vc.rev_topic_vocab,
            'senti_vocab': self.vc.senti_vocab,
            'rev_senti_vocab': self.vc.rev_senti_vocab
        }
        with open(self.cf.vocab_path, 'wb') as writer:
            pickle.dump(vocab, writer)

    def _load_word2vec(self):
        if not os.path.exists(self.cf.word2vec_path):
            return
        raw_word2vec = FastText.load_fasttext_format(self.cf.word2vec_path)
        reserved_tokens = self.reserved_token_for_dialog + self.reserved_token_for_gen
        oov_cnt = 0
        for vocab in self.vc.vocab:
            if vocab == '<pad>':
                vec = np.zeros(self.cf.word2vec_dim)
            elif vocab in reserved_tokens:
                vec = np.random.randn(self.cf.word2vec_dim) * 0.1
            else:
                if vocab in raw_word2vec:
                    vec = raw_word2vec[vocab]
                else:
                    oov_cnt += 1
                    vec = np.random.randn(self.cf.word2vec_dim) * 0.1
            self.word2vec.append(vec)
        print('word2vec cannot cover %f vocab.' % (float(oov_cnt)/len(self.vc.vocab)))

    def _convert_utt_to_id(self, utts):
        return [[self.vc.rev_vocab.get(token, self.vc.unk_id) for token in utt] for utt in utts]

    def get_utt_train_corpus(self):
        return self._convert_utt_to_id(self.train_corpus.utts)

    def get_utt_test_corpus(self):
        return self._convert_utt_to_id(self.test_corpus.utts)

    def get_utt_valid_corpus(self):
        return self._convert_utt_to_id(self.valid_corpus.utts)

    def _convert_dialog_to_id(self, dialogs):
        dialog_ids = []
        for dialog in dialogs:
            dialog_id = []
            for utt, caller, senti in dialog:
                utt_id = [self.vc.rev_vocab.get(token, self.vc.unk_id) for token in utt]
                senti_id = self.vc.rev_senti_vocab[senti] if senti is not None else None
                dialog_id.append((utt_id, caller, senti_id,))
            dialog_ids.append(dialog_id)
        return dialog_ids

    def get_dialog_train_corpus(self):
        return self._convert_dialog_to_id(self.train_corpus.dialog)

    def get_dialog_test_corpus(self):
        return self._convert_dialog_to_id(self.test_corpus.dialog)

    def get_dialog_valid_corpus(self):
        return self._convert_dialog_to_id(self.valid_corpus.dialog)

    def _convert_meta_to_id(self, meta):
        return [(vec_a_meta, vec_b_meta, self.vc.rev_topic_vocab[topic],)
                for vec_a_meta, vec_b_meta, topic in meta]

    def get_meta_train_corpus(self):
        return self._convert_meta_to_id(self.train_corpus.meta)

    def get_meta_test_corpus(self):
        return self._convert_meta_to_id(self.test_corpus.meta)

    def get_meta_valid_corpus(self):
        return self._convert_meta_to_id(self.valid_corpus.meta)
