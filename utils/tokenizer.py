from pathlib import Path

class CharacterTokenizer:
    def __init__(self, file_path):
        self.file_path = file_path
        with open(file_path,"r",encoding="utf-8") as f:
            self.text=f.read()

            self.chars = sorted(set(self.text))

            self.vocab_size=len(self.chars)

            self.char_to_idx = {
                ch:i
                for i, ch in enumerate(self.chars)
            }

            self.idx_to_char = {
                i:ch
                for i, ch in enumerate(self.chars)
            }

    def encode(self,text):
                return[
                    self.char_to_idx[c]
                    for c in text
                ]
    def decode(self,ids):
                return "".join(
                    self.idx_to_char[i]
                    for i in ids
                )