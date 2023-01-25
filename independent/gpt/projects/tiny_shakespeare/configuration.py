from independent.gpt.running.configuration import Configuration


def configuration() -> Configuration:
    return Configuration(
        vocabulary_size=5257,
        block_size=128,

        embedding_size=256,
        fully_connected_size=512,
        head_size=16,

        count_layers=4,
        count_heads=16,

        embedding_dropout=0.2,
        attention_dropout=0.2,
        projection_dropout=0.2,

        count_epochs=1101,
        training_iterations_per_epoch=150,
        validation_iterations_per_epoch=25,
        batch_size=64,
        learning_rate=1e-3,
        weight_decay=0.2,
        grad_norm_clip=1.0,
    )
