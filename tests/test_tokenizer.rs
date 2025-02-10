use toktokenizer::{
    preproc::{DefaultNormalizer, Normalize},
    BPETokenizer, Tokenizer,
};

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
