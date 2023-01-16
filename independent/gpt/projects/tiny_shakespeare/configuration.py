from independent.gpt.running.configuration import Configuration


config = Configuration(
    vocabulary_size=5257,
    block_size=32,

    embedding_size=32,
    fully_connected_size=64,
    head_size=8,

    count_layers=2,
    count_heads=4,

    embedding_dropout=0.1,
    attention_dropout=0.1,
    projection_dropout=0.1,

    count_epochs=1,
    batch_size=8,
    learning_rate=1e-3,
    weight_decay=0.2,
    grad_norm_clip=1.0,
)
