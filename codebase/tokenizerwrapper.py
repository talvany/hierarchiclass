from transformers import XLNetTokenizer
from codebase.settings import XLNET_BASE_PATH, VOCABULARY_PATH


class TokenizerWrapper():

    def __init__(self, model_folder):
        self.tokenizer = XLNetTokenizer(vocab_file=str(VOCABULARY_PATH), do_lower_case=False)

        self.CLS_ID = self.tokenizer.encode("<cls>")[0]
        self.SEP_ID = self.tokenizer.encode("<sep>")[0]

        self.tokenizer.save_vocabulary(model_folder)