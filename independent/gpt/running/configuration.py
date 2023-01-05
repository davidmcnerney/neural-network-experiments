class Configuration:
    def __init__(
            self,

            vocabulary_size: int,   # How many tokens in our vocabulary
            block_size: int,        # Maximum sequence length

            embedding_size: int,    # Size of embedding vector that represents a token
            fully_connected_size: int,
            head_size: int,

            count_layers: int,
            count_heads: int,       # number of attention heads in each layer

            embedding_dropout: float,
            attention_dropout: float,
            projection_dropout: float,
    ):
        self.vocabulary_size = vocabulary_size
        self.block_size = block_size

        self.embedding_size = embedding_size
        self.fully_connected_size = fully_connected_size
        self.head_size = head_size

        self.count_layers = count_layers
        self.count_heads = count_heads

        self.embedding_dropout = embedding_dropout
        self.attention_dropout = attention_dropout
        self.projection_dropout = projection_dropout

    @classmethod
    def standard(cls) -> "Configuration":
        return cls(
            vocabulary_size=50257,
            block_size=1024,

            embedding_size=512,
            fully_connected_size=2048,
            head_size=32,

            count_layers=8,
            count_heads=16,

            embedding_dropout=0.1,
            attention_dropout=0.1,
            projection_dropout=0.1,
        )
