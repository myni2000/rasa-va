from typing import Any, Dict, List, Text

import regex
import re

import rasa.shared.utils.io
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message

from pyvi import ViTokenizer, ViPosTagger

def Sorting(lst): 
    lst2 = sorted(lst, key=len) 
    return lst2


def find_entity(text):
    with open('rasa/nlu/tokenizers/entites.txt', "r", encoding = "utf-8") as f:
        entites = f.read()

    entites = entites.split("\n")
    entites = Sorting(entites)

    temp = None
    
    for entity in entites:
        if entity in text:
            temp = [text[0:text.index(entity)], entity, text[text.index(entity) + len(entity):]]
        try:
            temp.remove('')
            temp = [t.strip() for t in temp]
        except:
            pass
    if not (temp is None):
        return temp
    else:
        return [text]

class PyViVietnameseTokenizer(Tokenizer):

    defaults = {
        # Flag to check whether to split intents
        "intent_tokenization_flag": True,
        # Symbol on which intent should be split
        "intent_split_symbol": "_",
        # Regular expression to detect tokens
        "token_pattern": None,
    }

    # the following language should not be tokenized using the WhitespaceTokenizer
    not_supported_language_list = ["zh", "ja", "th"]

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        """Construct a new tokenizer using the WhitespaceTokenizer framework."""

        super().__init__(component_config)

        self.emoji_pattern = self.get_emoji_regex()

        if "case_sensitive" in self.component_config:
            rasa.shared.utils.io.raise_warning(
                "The option 'case_sensitive' was moved from the tokenizers to the "
                "featurizers.",
                docs=DOCS_URL_COMPONENTS,
            )

    @staticmethod
    def get_emoji_regex():
        return re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\u200d"  # zero width joiner
            "\u200c"  # zero width non-joiner
            "]+",
            flags=re.UNICODE,
        )

    def remove_emoji(self, text: Text) -> Text:
        """Remove emoji if the full text, aka token, matches the emoji regex."""
        match = self.emoji_pattern.fullmatch(text)

        if match is not None:
            return ""

        return text

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)

        # we need to use regex instead of re, because of
        # https://stackoverflow.com/questions/12746458/python-unicode-regular-expression-matching-failing-with-some-unicode-characters

        # remove 'not a word character' i
        
        texts = find_entity(text)

        words_total = []
        
        for t in texts:
          words = []
          for item in ViPosTagger.postagging(ViTokenizer.tokenize(t))[0]:
              words.append(" ".join(item.split("_")))
          words_total += words

        words_total = [self.remove_emoji(w) for w in words_total]
        words_total = [w for w in words_total if w]

        # if we removed everything like smiles `:)`, use the whole text as 1 token
        if not words_total:
            words_total = [text]

        tokens = self._convert_words_to_tokens(words_total, text)

        return self._apply_token_pattern(tokens)