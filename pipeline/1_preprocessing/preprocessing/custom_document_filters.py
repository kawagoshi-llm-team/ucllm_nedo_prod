from hojichar import document_filters, Document
from fugashi import Tagger
from ftlangdetect import detect


from os import PathLike
from typing import Any, Union
import re

tagger = Tagger('-Owakati')


class DiscardAdultContentJa(document_filters.NgWordsFilterJa):
    """
    TokenFilter の実装例です.
    日本語の成人向けコンテンツを閾値に応じて排除します.
    """

    def __init__(
        self,
        dict_path: Union[str, PathLike] = document_filters.BASE_PATH / "dict/adult_keywords_ja.txt",
        threshold: float = 0.01,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(dict_path, *args, **kwargs)
        self.threshold = threshold

    def apply(self, doc: Document) -> Document:
        adult_keywords_pattern = self.keyword_pat
        matches = re.findall(adult_keywords_pattern, doc.text)
        adult_content_count = len(matches)
        total_words_count = len(tagger.parse(doc.text).split()) # Owakatiで分かち書きして単語数を数える

        if total_words_count > 0 and adult_content_count / total_words_count > self.threshold:
            doc.is_rejected = True # adult keywordsの割合が閾値を超えたらreject

        return doc
    
class SelectJapanese(document_filters.AcceptJapanese):
    """
    TokenFilter の実装例です.
    日本語以外の文書を排除します.
    """

    def __init__(self, lookup_size, *args: Any, **kwargs: Any) -> None:
        super().__init__(lang="ja", lookup_size=lookup_size, *args, **kwargs)

    def apply(self, doc: Document) -> Document:
        filterd_doc = super().apply(doc) # テキストを左から`lookup_size` (デフォルトで50字) 参照し, ひらがな・カタカナが存在すれば日本語と判定す
        if filterd_doc.is_rejected is False:
   
            result = detect(text=filterd_doc.text, low_memory=False) # fasttextを用いて言語判定
            if result["lang"] != "ja" or result["score"] < 0.9: # fasttextのスコアが0.9未満の場合はreject
                filterd_doc.is_rejected = True

        return filterd_doc