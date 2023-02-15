import random
import unicodedata
from pathlib import Path

from classopt import classopt
from more_itertools import divide, flatten
from tqdm import tqdm

import src.utils as utils

import pandas as pd


@classopt(default_long=True)
class Args:
    input_dir: Path = "./data_rcqa"
    output_dir: Path = "./datasets/rcqa"
    seed: int = 42


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


def main(args: Args):
    random.seed(args.seed)

    labels = set()

    df = pd.read_json(args.input_dir / 'all-v1.0.json', orient='records', lines=True)
    # データフォーマット
    # qid: int
    # competotion: str
    # timestamp: timestamp
    # format: str 
    # question: str 
    # answer: str 
    # documents: list
    #   title: text
    #   text: text
    #   score: int


    portions = list(divide(10, df.iterrows()))
    train, val, test = list(flatten(portions[:-2])), portions[-2], portions[-1]
    # qidごとにtrain/val/test分割する
    # documentsまで展開しては分割しない

    for n, d in zip(['train','val','test'], [train,val, test]):
        data = []
        for i, record in d:
            a = record.copy()
            a.pop('documents')
            for j, k in enumerate(record['documents']):
                data.append({
                    **a,
                    **k,
                    "did": j+1  # (qid, did) で一意性を担保
                })

        random.shuffle(data)
        data = pd.DataFrame(data)

        data.text = data.text.apply(process_title)
        data.title = data.title.apply(process_title)
        data.question = data.question.apply(process_title)
        data.answer = data.answer.apply(process_title)

        data.timestamp = data.timestamp.apply(lambda x: f'{x}'.split(' ')[0])

        data.to_json(args.output_dir / f"{n}.jsonl", orient='records', force_ascii=False, lines=True)

        for i in data.score.unique():
            labels.add(i)

    label2id = {f"{label}": i for i, label in enumerate(sorted(labels))}

    utils.save_json(label2id, args.output_dir / "label2id.json")


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
