from textwrap import dedent

def test_build_vocab():
    training_text = dedent("""
        Big jug dig dug, hello.
    """)

    expected_vocab = {
        0: ""
    }
