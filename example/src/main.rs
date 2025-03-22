use toktkn::{BPETokenizer, Pretrained, Tokenizer, TokenizerConfig};

fn main() {
    let config = TokenizerConfig::new(10, None);
    let mut tok = BPETokenizer::new(config);

    let corpus: String = "this is ideally some useful text used to train your tokenizer".into();

    tok.train(&corpus);
    assert_eq!(tok.len(), 10);

    let test_sentence = "testing testing 123".to_string();
    assert_eq!(tok.decode(&tok.encode(&test_sentence)), test_sentence);

    // save to disk
    tok.save_pretrained("tokenizer.json")
        .expect("failed to save tokenizer!");

    // ... and load
    let new_tok =
        BPETokenizer::from_pretrained("tokenizer.json").expect("failed to load tokenizer");
    assert_eq!(new_tok.len(), tok.len());
}
