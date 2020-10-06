import torch
import numpy as np
from torch.utils.data import Dataset


class KGCVAEDataset(Dataset):
    pad_token_idx = 0
    data = []

    def __init__(self, name, dialog, meta, config):
        assert len(dialog) == len(meta)

        class Config:
            def __init__(self):
                self.name = name
                self.utt_per_case = config['utt_per_case']
                self.max_utt_len = config['max_utt_len']
                self.is_inference = config.get('inference', False)

        self.cf = Config()

        self.dialog_lengths = [len(data) for data in dialog]
        print('Max len %d and Min len %d and Avg len %f' % (
            np.max(self.dialog_lengths),
            np.min(self.dialog_lengths),
            float(np.mean(self.dialog_lengths))
        ))

        self.indexes = list(np.argsort(self.dialog_lengths))

        for idx, data in enumerate(dialog):
            data_length = len(data)
            vec_a_meta, vec_b_meta, topic = meta[idx]

            topics = torch.LongTensor([topic])

            if self.cf.is_inference:
                end_idx_offset_start = data_length
                end_idx_offset_end = data_length + 1
            else:
                end_idx_offset_start = 2
                end_idx_offset_end = data_length

            for end_idx_offset in range(end_idx_offset_start, end_idx_offset_end):
                start_idx = max(0, end_idx_offset - self.cf.utt_per_case)
                end_idx = end_idx_offset

                cut_row = data[start_idx:end_idx]
                in_row = cut_row[0:-1]
                out_row = cut_row[-1]

                out_utt, out_caller, out_senti = out_row

                if out_caller == 1:
                    my_profile = torch.FloatTensor(vec_b_meta)
                    ot_profile = torch.FloatTensor(vec_a_meta)
                else:
                    my_profile = torch.FloatTensor(vec_a_meta)
                    ot_profile = torch.FloatTensor(vec_b_meta)

                context_lens = torch.FloatTensor([len(cut_row) - 1])

                padded_utt_pairs = [self.get_sliced_or_padded_sentence(utt) for utt in in_row]

                context_utts = np.zeros((self.cf.utt_per_case, self.cf.max_utt_len))
                context_utts[:len(in_row)] = [utt_pair[0] for utt_pair in padded_utt_pairs]
                context_utts = torch.LongTensor(context_utts)

                in_row_lens = np.zeros(self.cf.utt_per_case)
                in_row_lens[:len(in_row)] = [utt_pair[1] for utt_pair in padded_utt_pairs]

                floors = np.zeros(self.cf.utt_per_case)
                floors[:len(in_row)] = [int(caller == out_caller) for utt, caller, senti in in_row]
                floors = torch.LongTensor(floors)

                padded_out_utt = self.get_sliced_or_padded_sentence(out_utt)
                out_utts = torch.LongTensor(padded_out_utt[0])
                out_lens = torch.LongTensor(padded_out_utt[1])

                out_floor = torch.LongTensor(out_caller)
                out_das = torch.LongTensor(out_senti)

                self.data.append({
                    'topics': topics,
                    'my_profile': my_profile,
                    'ot_profile': ot_profile,
                    'context_lens': context_lens,
                    'context_utts': context_utts,
                    'floors': floors,
                    'out_utts': out_utts,
                    'out_lens': out_lens,
                    'out_floor': out_floor,
                    'out_das': out_das
                })

    def get_sliced_or_padded_sentence(self, sentence, do_pad=True):
        if len(sentence) >= self.cf.max_utt_len:
            return sentence[:self.cf.max_utt_len - 1] + [sentence[-1]], self.cf.max_utt_len
        elif do_pad:
            return sentence + [self.pad_token_idx] * (self.cf.max_utt_len - len(sentence)), len(sentence)
        else:
            return sentence, len(sentence)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
