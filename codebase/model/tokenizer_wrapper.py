from transformers import XLNetTokenizer
from codebase.settings import XLNET_BASE_PATH, VOCABULARY_PATH


class TokenizerWrapperSingleton:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if TokenizerWrapperSingleton.__instance == None:
            TokenizerWrapperSingleton()
        return TokenizerWrapperSingleton.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if TokenizerWrapperSingleton.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            self.tokenizer = XLNetTokenizer(
                vocab_file=str(VOCABULARY_PATH), do_lower_case=False
            )

            self.CLS_ID = self.tokenizer.encode("<cls>")[0]
            self.SEP_ID = self.tokenizer.encode("<sep>")[0]

            TokenizerWrapperSingleton.__instance = self
