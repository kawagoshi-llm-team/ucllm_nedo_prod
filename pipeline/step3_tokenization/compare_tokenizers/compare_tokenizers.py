import argparse
import asyncio
import aiohttp
import yaml
import polars as pl
import numpy as np
from typing import List, Optional, Dict


# コマンドライン引数を解析する関数
def parse_args():
    parser = argparse.ArgumentParser(
        description="YAMLファイルからモデル名を取得し、Hugging Faceから設定を取得する"
    )
    parser.add_argument(
        "--yaml_path", type=str, required=True, help="YAMLファイルのパス"
    )
    return parser.parse_args()


# YAMLファイルからモデル名のリストを取得する関数
async def get_model_names(yaml_path: str) -> List[str]:
    with open(yaml_path, "r") as file:
        yaml_data = yaml.safe_load(file)
    return yaml_data["model"]


# URLを作成する関数
def create_url(model_name: str) -> str:
    return f"https://huggingface.co/{model_name}/raw/main/config.json"


# JSONファイルから必要な情報を取得する関数
async def fetch_model_info(
    session: aiohttp.ClientSession, model_name: str
) -> Dict[str, Optional[str]]:
    url = create_url(model_name)
    async with session.get(url) as response:
        if response.status == 200:
            json_data = await response.json(content_type=None)
            return {
                "model_name": model_name,
                "architectures": json_data.get("architectures", ["NaN"])[0],
                "model_type": json_data.get("model_type", "NaN"),
                "vocab_size": json_data.get("vocab_size", None),  # NaNをNoneに変更
                "tokenizer_class": json_data.get("tokenizer_class", "NaN"),
                "URL": url,
            }
        else:
            return {
                "model_name": model_name,
                "architectures": "NaN",
                "model_type": "NaN",
                "vocab_size": None,  # NaNをNoneに変更
                "tokenizer_class": "NaN",
                "URL": f"https://huggingface.co/{model_name}/blob/main/README.md",
            }


# モデル情報を非同期で取得する関数
async def fetch_all_model_infos(
    model_names: List[str],
) -> List[Dict[str, Optional[str]]]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_model_info(session, model_name) for model_name in model_names]
        return await asyncio.gather(*tasks)


# メインの処理を行う関数
async def main():
    args = parse_args()
    model_names = await get_model_names(args.yaml_path)
    model_infos = await fetch_all_model_infos(model_names)

    # polarsデータフレームに格納
    df = pl.DataFrame(model_infos)
    df = df.with_columns(
        [pl.col("vocab_size").cast(pl.Int64, strict=False)]  # より柔軟なキャストを行う
    )

    # vocab_sizeが大きい順にソートする
    df = df.sort("vocab_size", descending=True)

    # csvファイルとして出力する
    df.write_csv("pipeline/step3_tokenization/compare_tokenizers/model_info.csv")

    # markdown形式でコンソール上に出力する
    with pl.Config(tbl_rows=df.select(pl.len()).item()):
        print(df)


# スクリプトが直接実行された場合にmain関数を実行する
if __name__ == "__main__":
    asyncio.run(main())
