import random
import unicodedata
from pathlib import Path

from classopt import classopt
from more_itertools import divide, flatten
from tqdm import tqdm

import src.utils as utils

import pandas as pd


import numpy as np

from transformers import BertJapaneseTokenizer, BertModel
import torch

import joblib

# https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2
class SentenceBertJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
        self.model = BertModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", 
                                           truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        # return torch.stack(all_embeddings).numpy()
        return torch.stack(all_embeddings)

"""
MODEL_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"  # <- v2です。
model = SentenceBertJapanese(MODEL_NAME)

sentences = ["暴走したAI", "暴走した人工知能"]
sentence_embeddings = model.encode(sentences, batch_size=8)

print("Sentence embeddings:", sentence_embeddings)
"""


@classopt(default_long=True)
class Args:
    input_dir: Path = "./datasets/rcqa"
    output_dir: Path = "./datasets/rcqa"
    seed: int = 42
    model_name: str = "sonoisa/sentence-bert-base-ja-mean-tokens-v2"


def process_title(title: str) -> str:
    title = unicodedata.normalize("NFKC", title)
    title = title.strip("　").strip()
    return title


# 記事本文の前処理
# 重複した改行の削除、文頭の全角スペースの削除、NFKC正規化を実施
def process_body(body: list[str]) -> str:
    body = [unicodedata.normalize("NFKC", line) for line in body]
    body = [line.strip("　").strip() for line in body]
    body = [line for line in body if line]
    body = "\n".join(body)
    return body


# QA, passageの埋め込みとスコアを抽出して保存
def main(args: Args):
    random.seed(args.seed)

    model = SentenceBertJapanese(args.model_name)

    for name in ['train','val','test']:
        print(f'processing .. {name}')

        df = pd.read_json(args.input_dir / f"{name}.jsonl", orient='records', lines=True)

        qa = df['question'] + "答えは" + df['answer']   # "...は何でしょう?答えは...."
        passage = df['text']
        score = df['score'].values

        n = 40
        batch_size = 4

        qa_data = [ qa[idx: idx + n] for idx in range(0,len(qa), n)]
        qa_emb = np.vstack([model.encode(data, batch_size=batch_size).cpu().detach().numpy() for data in tqdm(qa_data)])

        joblib.dump(qa_emb, args.output_dir / f"{name}_qa_emb.npy.pickle", compress=3)
        del qa_data, qa_emb

        passage_data = [ passage[idx: idx + n] for idx in range(0,len(passage), n)]
        passage_emb = np.vstack([model.encode(data, batch_size=batch_size).cpu().detach().numpy() for data in tqdm(passage_data)])

        joblib.dump(passage_data, args.output_dir / f"{name}_passage_emb.npy.pickle", compress=3)
        del passage_data, passage_emb

        score = np.expand_dims(score.astype('float32'), axis=1)

        joblib.dump(score, args.output_dir / f"{name}_rcqa_score.npy.pickle", compress=3)
        del score

        # これを用いて qa_emb + passage_emb -> score 


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
