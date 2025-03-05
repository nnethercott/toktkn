use super::*;
extern crate tempdir;
use tempdir::TempDir;
use rstest::*;

use toktokenizer::{config::TokenizerConfig, preproc::Normalizer, BPETokenizer, Pretrained};

// some test helpers
use helpers::{tmpdir, tokenizer};

#[rstest]
fn test_serialize_config(tmpdir: &TempDir) -> std::io::Result<()> {
    // arrange
    let file_path = tmpdir.path().join("config.txt");
    let mut config = TokenizerConfig::new(42, None);

    // act
    config.save_pretrained(&file_path)?;
    assert!(file_path.exists());

    config = TokenizerConfig::from_pretrained(&file_path)?;
    assert_eq!(config.vocab_size, 42);
    assert_eq!(config.preproc, Normalizer::default());

    Ok(())
}

#[rstest]
fn test_serialize_tokenizer(tmpdir: &TempDir, tokenizer: &BPETokenizer) -> std::io::Result<()> {
    // arrange
    let file_path = tmpdir.path().join("tokenizer.json");

    // act
    tokenizer.save_pretrained(&file_path)?;
    assert!(file_path.exists());

    let new_tokenizer = BPETokenizer::from_pretrained(&file_path)?;
    assert_eq!(new_tokenizer.config.vocab_size, tokenizer.config.vocab_size);
    assert_eq!(new_tokenizer.encoder, tokenizer.encoder);

    Ok(())
}
