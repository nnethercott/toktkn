use rustc_hash::{FxBuildHasher, FxHashMap, FxHasher};
use toktokenizer::{
    config::TokenizerConfig, preproc::{DefaultNormalizer, Normalize, DEFAULT_NORMALIZER}, BPETokenizer, Tokenizer
};

use crate::helpers::{get_corpus, get_sentence};

#[test]
fn test_encode_decode() {
    let config = TokenizerConfig::new(42);
    let mut tok = BPETokenizer::new(config);

    tok.train(&get_corpus());

    let text = get_sentence();
    let encoded = tok.encode(&text);

    assert!(text.len() >= encoded.len(),);
    assert_eq!(text, tok.decode(&encoded),);
}

#[test]
fn test_restart_train_works() {
    let normalizer = DefaultNormalizer;
    let mut tok = BPETokenizer::new(&normalizer);

    let corpus = get_corpus();

    tok.train(&corpus, 50);
    assert!(tok.len() == 50);

    tok.train(&corpus, 75);
    assert!(tok.len() == 75);
}

#[test]
fn test_add_special_tokens() {
    let mut tok = BPETokenizer::new(&DEFAULT_NORMALIZER);

    // add special tokens before train
    let special_tokens = vec!["<s>", "hello", "world", "</s>"];
    tok.add_special_tokens(special_tokens);

    assert_eq!(tok.len(), 4);
    assert_eq!(
        tok.special_tokens_map,
        Some(FxHashMap::from_iter(vec![
            ("<s>".to_string(), 128),
            ("hello".to_string(), 129),
            ("world".to_string(), 130),
            ("</s>".to_string(), 131)
        ]))
    );

    let corpus = get_corpus();
    tok.train(&corpus, 10);
    assert_eq!(tok.len(), 10);

    dbg!(tok.encoder);
    dbg!(tok.special_tokens_map);
}

#[test]
fn test_special_tokens_doesnt_break_encoding() {
    let mut tok = BPETokenizer::new(&DEFAULT_NORMALIZER);
    let special_tokens = vec!["<s>", "hello", "world", "</s>"];
    tok.add_special_tokens(special_tokens);

    let corpus = get_corpus();
    tok.train(&corpus, 10);
    let special_tokens = vec!["<nate>"];

    let mut sample = String::from("hello hello world <s></s> some more text goes here");
    sample += &get_sentence();
    assert_eq!(tok.decode(&tok.encode(&sample)), sample);

    dbg!(tok.encode(&sample));
}
