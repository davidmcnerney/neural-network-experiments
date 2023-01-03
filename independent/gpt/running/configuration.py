class Configuration:
    def __init__(self):
        self.vocabulary_size = 50257    # How many tokens in our vocabulary
        self.block_size = 1024          # Maximum sequence length
        self.embedding_size = 512       # Size of embedding vector that represents a token
