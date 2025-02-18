use rustc_hash::{FxBuildHasher, FxHashMap, FxHasher};
use toktokenizer::{
    config::TokenizerConfig, preproc::{DefaultNormalizer, Normalize}, BPETokenizer, Tokenizer
};

use crate::helpers::{get_corpus, get_sentence};

#[test]
fn test_encode_decode() {
    let config = TokenizerConfig::new(42, None);
    let mut tok = BPETokenizer::new(config);

    tok.train(&get_corpus());

    let text = get_sentence();
    let encoded = tok.encode(&text);

    assert!(text.len() >= encoded.len(),);
    assert_eq!(text, tok.decode(&encoded),);
}

#[test]
fn test_train_works() {
    let config = TokenizerConfig::new(42, None);
    let mut tok = BPETokenizer::new(config);

    let corpus = get_corpus();

    tok.train(&corpus);
    assert!(tok.len() == tok.config.vocab_size);
}

#[test]
fn test_special_tokens_map(){
    let config = TokenizerConfig::new(10, None);
    let mut tok = BPETokenizer::new(config);

    // add special tokens before train
    let special_tokens = vec!["<s>", "hello", "world", "</s>"];
    tok.add_special_tokens(special_tokens);

    assert_eq!(tok.len(), 4);
    assert_eq!(
        tok.config.special_tokens_map,
        Some(FxHashMap::from_iter(vec![
            ("<s>".to_string(), 128),
            ("hello".to_string(), 129),
            ("world".to_string(), 130),
            ("</s>".to_string(), 131)
        ]))
    );

    let corpus = get_corpus();
    tok.train(&corpus);
    assert_eq!(tok.len(), 10);

    dbg!(tok.encoder);
    dbg!(tok.config.special_tokens_map);
}

#[test]
fn test_special_tokens_doesnt_break_encoding() {
    let config = TokenizerConfig::new(10, None);
    let mut tok = BPETokenizer::new(config);

    let special_tokens = vec!["<s>", "hello", "world", "</s>"];
    tok.add_special_tokens(special_tokens);

    let corpus = get_corpus();
    tok.train(&corpus);
    let special_tokens = vec!["<nate>"];

    let mut sample = String::from("hello hello world <s></s> some more text goes here");
    sample += &get_sentence();
    assert_eq!(tok.decode(&tok.encode(&sample)), sample);

    dbg!(tok.encode(&sample));
}
