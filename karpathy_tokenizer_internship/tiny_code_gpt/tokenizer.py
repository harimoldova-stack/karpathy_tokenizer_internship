class CharacterTokenizer:
    def __init__(self, text):
        # Find every unique character
        self.chars = sorted(list(set(text)))

        # Character -> number
        self.stoi = {
            character: index
            for index, character in enumerate(self.chars)
        }

        # Number -> character
        self.itos = {
            index: character
            for index, character in enumerate(self.chars)
        }

        self.vocab_size = len(self.chars)

    def encode(self, text):
        """Convert text into a list of numbers."""
        return [self.stoi[character] for character in text]

    def decode(self, numbers):
        """Convert numbers back into text."""
        return "".join(self.itos[number] for number in numbers)