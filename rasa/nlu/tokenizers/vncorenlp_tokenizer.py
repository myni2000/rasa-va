import re
from typing import Any, Dict, List, Text
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.shared.nlu.training_data.message import Message

from rasa.nlu.constants import TOKENS_NAMES, MESSAGE_ATTRIBUTES
from vncorenlp import VnCoreNLP

rdrsegmenter = VnCoreNLP("rasa/nlu/tokenizers/vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')
class VncoreNLPVietnameseTokenizer(Tokenizer):

    provides = [TOKENS_NAMES[attribute] for attribute in MESSAGE_ATTRIBUTES]

    def __init__(self, component_config: Dict[Text, Any] = None) -> None:
        super().__init__(component_config)

    def tokenize(self, message: Message, attribute: Text) -> List[Token]:
        text = message.get(attribute)
        sentences = rdrsegmenter.tokenize(text)[0]

        words = []
        for item in sentences:
            words.append(" ".join(item.split("_")))

        return self._convert_words_to_tokens(words, text)