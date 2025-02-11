use toktokenizer::{
    preproc::{DefaultNormalizer, Normalize},
    BPETokenizer, Tokenizer,
};

use fake::faker::lorem::en::{Paragraph, Sentence};
use fake::{Fake, Faker};

#[test]
fn test_encode_decode() {
    let mut tok = BPETokenizer::new(&DefaultNormalizer);
    let _ = tok.load_encoder("wikibpe.json");

    let text = String::from("this is an example. 😎");
    assert_eq!(
        text,
        tok.decode(&tok.encode(&text)),
        "tokenize/detokenize should be inverse operations"
    );
}

#[test]
fn test_compression() {
    let mut tok = BPETokenizer::new(&DefaultNormalizer);
    let _ = tok.load_encoder("wikibpe.json");

    let text = String::from("this is an example. 😎");

    assert!(
        text.len() >= tok.encode(&text).len(),
        "tokenized sequence should be shorter or equal to original"
    );
}

#[test]
fn test_restart_train_works(){
    let normalizer = DefaultNormalizer;
    let mut tok = BPETokenizer::new(&normalizer);

    let corpus: String = Paragraph(100..101).fake();

    tok.train(&corpus, 50);
    assert!(tok.len()==50, "encoder size does not match vocab_size");

    tok.train(&corpus, 75);
    assert!(tok.len()==75, "encoder size does not match vocab_size");
}

