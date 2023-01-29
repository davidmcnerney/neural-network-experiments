from gpt.running.configuration import Configuration


def configuration() -> Configuration:
    return Configuration(
        vocabulary_size=5257,
        block_size=128,

        embedding_size=512,
        fully_connected_size=1024,
        head_size=32,

        count_layers=16,
        count_heads=16,

        embedding_dropout=0.2,
        attention_dropout=0.2,
        projection_dropout=0.2,

        count_epochs=719,
        training_iterations_per_epoch=150,
        validation_iterations_per_epoch=25,
        batch_size=64,

        learning_rate=1e-3,
        count_warmup_iterations=100,
        count_decay_iterations=4900,
        decayed_learning_rate=1e-4,
        adam_beta_1=0.9,
        adam_beta_2=0.99,
        weight_decay=0.2,
        grad_norm_clip=1.0,
    )
