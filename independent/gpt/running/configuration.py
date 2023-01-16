class Configuration:
    def __init__(
            self,

            # Vocabulary and tokens
            vocabulary_size: int,   # How many tokens in our vocabulary
            block_size: int,        # Maximum sequence length

            # Dimensionalities
            embedding_size: int,    # Size of embedding vector that represents a token
            fully_connected_size: int,
            head_size: int,

            # Network size
            count_layers: int,
            count_heads: int,       # number of attention heads in each layer

            # Dropout
            embedding_dropout: float,
            attention_dropout: float,
            projection_dropout: float,

            # Training
            count_epochs: int,
            batch_size: int,
            learning_rate: float,
            weight_decay: float,    # only applied to some parameters; see training.py
            grad_norm_clip: float,
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

        self.count_epochs = count_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_norm_clip = grad_norm_clip

    # TODO: revisit all these parameters

    @classmethod
    def for_tests(cls) -> "Configuration":
        return cls(
            vocabulary_size=1536,
            block_size=256,

            embedding_size=48,
            fully_connected_size=192,
            head_size=16,

            count_layers=3,
            count_heads=3,

            embedding_dropout=0.1,
            attention_dropout=0.1,
            projection_dropout=0.1,

            count_epochs=2,
            batch_size=64,
            learning_rate=1e-3,
            weight_decay=0.2,
            grad_norm_clip=1.0,
        )

    @classmethod
    def standard(cls) -> "Configuration":
        return cls(
            vocabulary_size=50257,
            block_size=1024,

            embedding_size=768,
            fully_connected_size=1536,
            head_size=64,

            count_layers=12,
            count_heads=12,

            embedding_dropout=0.1,
            attention_dropout=0.1,
            projection_dropout=0.1,

            count_epochs=1,
            batch_size=8,
            learning_rate=1e-3,
            weight_decay=0.2,       
            grad_norm_clip=1.0,
        )
