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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

corpus_config_path = 'config/cvae_corpus_kor.json'
dataset_config_path = 'config/cvae_dataset_kor.json'
trainer_config_path = 'config/cvae_trainer_kor.json'
model_config_path = 'config/cvae_model_kor.json'

model_path = "your_model_path"


def main_code():
    pass


if __name__ == "__main__":
    main_code()
