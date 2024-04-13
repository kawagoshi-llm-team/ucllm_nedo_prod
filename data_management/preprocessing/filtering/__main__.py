import argparse
import json
import os
from hojichar import document_filters, tokenization, Compose, Document
from preprocessing.filtering import custom_token_filters, custom_tokenization, custom_document_filters

def process_json_line(line: str, cleaner, writer, rejected_writer):
    result = cleaner.apply(Document(line))
    if result.is_rejected:
        # 拒否された行を拒否専用のファイルに保存
        rejected_writer.write(json.dumps({'text': line}, ensure_ascii=False) + "\n")
        return False
    else:
        # 拒否されていない行を通常のファイルに保存
        writer.write(result.text + "\n")
        return True

def __readline(input_file: str, start_line=0):
    line_count = 0
    with open(input_file, "r") as fp:
        for line in fp:
            if line_count >= start_line:
                yield line.strip()
            line_count += 1

def get_progress_file_path(output_base, input_file_prefix):
    return os.path.join(output_base, f"{input_file_prefix}_progress.txt")

def save_progress(output_base, input_file_prefix, line_num):
    progress_file_path = get_progress_file_path(output_base, input_file_prefix)
    with open(progress_file_path, "w") as file:
        file.write(str(line_num))

def load_progress(output_base, input_file_prefix):
    progress_file_path = get_progress_file_path(output_base, input_file_prefix)
    if os.path.exists(progress_file_path):
        with open(progress_file_path, "r") as file:
            return int(file.read().strip())
    return 0

def filtering(input_dir: str, output_base: str):
    os.makedirs(output_base, exist_ok=True)
    input_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".jsonl")]

    cleaner = Compose([
        document_filters.JSONLoader(), # テキストをJsonとして解釈し,`key`で指定した要素を文字列としてdoumentに格納.デフォルト`key`は'text'.
        document_filters.DocumentNormalizer(), # 文書の正規化を行う.デフォルトでは全角文字を半角文字に変換する.(NFKC)
        document_filters.DiscardBBSComments(), # 正規表現 "BBS Patern" に `max_allow_num`(default: 14) 回よりたくさんマッチする文書を破棄。
        document_filters.DiscardAds(), # 主に広告キーワードを`max_allow_num`より多く含む文書を破棄.
        document_filters.DiscardDiscriminationContentJa(), # 日本語の差別キーワード(および不適切語)を含む文書を破棄.
        document_filters.DocumentLengthFilter(min_doc_len=10, max_doc_len=100000), # 文書の長さがmin_doc_lenより短い、またはmax_doc_lenより長い文書を破棄.
        custom_document_filters.DiscardAdultContentJa(), # 日本語の成人向けコンテンツを閾値に応じて排除.
        custom_tokenization.NewLineSentenceTokenizer(), # 改行文字で文章を区切る.
        custom_token_filters.RemoveOneword(), # 1単語のみ以下のパターンを削除.
        custom_tokenization.MergeTokens(delimiter="\n"), # 破棄されていないトークンをdelimeterで結合.
        #custom_tokenization.WakatiTokenizer(), # fugashi を用いて文を分割.
        custom_token_filters.RemoveDate(), # 日付のみのパターンを削除.
        tokenization.MergeTokens(), # 破棄されていないトークンを結合.
        #custom_document_filters.SelectJapanese(lookup_size=50), # 日本語以外の文書を排除.
        document_filters.MaskPersonalInformation(), # キュメントに含まれる電話番号・電子メールアドレスを一部マスキング.
        document_filters.JSONDumper(dump_reason=False), # dump_reason=Tureの場合、ドキュメントの破棄事由をJSON形式で付与して出力. JSON形式で出力するため、最後に実行する.（デバッグ用）
    ])

    for input_file in input_files:
        print("cleaning: " + input_file)
        input_file_prefix = os.path.splitext(os.path.basename(input_file))[0]
        output_file_path = os.path.join(output_base, f"{input_file_prefix}.jsonl")
        rejected_file_path = os.path.join(output_base, f"{input_file_prefix}_rejected.jsonl")
        progress = load_progress(output_base, input_file_prefix)

        with open(output_file_path, "a" if progress else "w") as writer, open(rejected_file_path, "a" if progress else "w") as rejected_writer:
            for line_num, line in enumerate(__readline(input_file, progress), start=progress):
                process_json_line(line, cleaner, writer, rejected_writer)
                save_progress(output_base, input_file_prefix, line_num + 1)

def main():
    parser = argparse.ArgumentParser(description='Process some documents.')
    parser.add_argument('--input_dir', type=str, help='The input directory containing documents to process', required=True)
    parser.add_argument('--output_dir', type=str, help='The output directory to save processed documents', required=True)
    args = parser.parse_args()

    filtering(input_dir=args.input_dir, output_base=args.output_dir)

if __name__ == "__main__":
    main()

