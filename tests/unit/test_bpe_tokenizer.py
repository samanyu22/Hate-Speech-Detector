import pytest
from pathlib import Path
from src.preprocessing.text.bpe_tokenizer import BytePairTokenizer

@pytest.fixture
def trained_bpe():
    texts = [
        "These illegal immigrants are vermin infecting our country and should be eliminated",
        "In historical archives propaganda used terms like subhuman to dehumanize",
        "We must defend human rights and build a community founded on equality and dignity",
        "ye log desh ke gaddar hai inko bahar phenko",
        "stinky dirty unclean people should be avoided",
        "हमारे देश की असली ताकत इसकी विविधता और भाईचारे में है"
    ]
    bpe = BytePairTokenizer()
    bpe.train(texts, target_vocab_size=200, min_frequency=1)
    return bpe

def test_bpe_subword_tokenization_known_word(trained_bpe):
    """Test that high-frequency words are merged into cohesive subword units."""
    subwords = trained_bpe.tokenize_word("vermin")
    assert len(subwords) >= 1
    assert "</w>" in subwords[-1]

def test_bpe_oov_subword_decomposition_not_unk(trained_bpe):
    """
    Core Test: Out-of-Vocabulary words (e.g. 'stiiinky', 'gaddaron') must NOT
    collapse into <UNK>. They must decompose into constituent subwords/characters.
    """
    # 'stiiinky' was never in the training text
    oov_word = "stiiinky"
    subwords = trained_bpe.tokenize_word(oov_word)
    
    # Subwords should preserve the characters/stems and not return ['<UNK>']
    assert "<UNK>" not in subwords
    assert len(subwords) > 1
    # Check that decoding reconstructs the original word
    token_ids = trained_bpe.encode(oov_word)
    assert 1 not in token_ids # 1 is <UNK> ID
    decoded = trained_bpe.decode(token_ids)
    assert decoded == oov_word

def test_bpe_multilingual_hindi_oov(trained_bpe):
    """Test that novel Hindi/Hinglish morphological extensions are decomposed cleanly."""
    oov_hinglish = "gaddari"
    subwords = trained_bpe.tokenize_word(oov_hinglish)
    assert "<UNK>" not in subwords
    token_ids = trained_bpe.encode(oov_hinglish)
    decoded = trained_bpe.decode(token_ids)
    assert decoded == oov_hinglish

def test_bpe_save_and_load(tmp_path, trained_bpe):
    """Verify that learned merges and vocabulary can be serialized and reloaded."""
    save_file = tmp_path / "bpe_model.json"
    trained_bpe.save(str(save_file))
    
    loaded_bpe = BytePairTokenizer(str(save_file))
    assert loaded_bpe.is_trained
    assert len(loaded_bpe.merges) == len(trained_bpe.merges)
    assert len(loaded_bpe.vocab) == len(trained_bpe.vocab)
    
    test_text = "equality for all"
    assert trained_bpe.encode(test_text) == loaded_bpe.encode(test_text)
