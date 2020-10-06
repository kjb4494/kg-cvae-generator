"""
Copyright 2019 Pingpong AI Research, ScatterLab

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import utils
from data_processing.corpus import KGCVAECorpus
from data_processing.dataset import KGCVAEDataset

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

corpus_config_path = 'config_files/cvae_corpus.json'
dataset_config_path = 'config_files/cvae_dataset.json'
trainer_config_path = 'config_files/cvae_trainer.json'
model_config_path = 'config_files/cvae_model.json'

overall = {
    "work_dir": "./work",
    "log_dir": "log",
    "model_dir": "weights",
    "test_dir": "test"
}


def main_code():
    corpus_config = utils.load_config(corpus_config_path)
    corpus = KGCVAECorpus(config=corpus_config)

    dataset_config = utils.load_config(dataset_config_path)

    train_set = KGCVAEDataset(
        name='Train',
        dialog=corpus.get_dialog_train_corpus(),
        meta=corpus.get_meta_train_corpus(),
        config=dataset_config
    )


if __name__ == "__main__":
    main_code()
