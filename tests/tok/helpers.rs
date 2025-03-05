use fake::faker::lorem::en::{Paragraph, Sentence};
use fake::Fake;
use rstest::fixture;
use tempdir::TempDir;
use toktokenizer::config::TokenizerConfig;
use toktokenizer::BPETokenizer;
use uuid::Uuid;

pub fn get_corpus() -> String {
    let corpus: String = Paragraph(100..201).fake();
    corpus
}

pub fn get_sentence() -> String {
    let sentence: String = Sentence(10..50).fake();
    sentence
}

#[fixture]
#[once]
pub fn tokenizer() -> BPETokenizer{
    let config = TokenizerConfig::new(42, None);
    let mut tok = BPETokenizer::new(config);
    tok.train(&get_corpus());
    tok
}


#[fixture]
#[once]
pub fn tmpdir()->TempDir{
    let dir = Uuid::new_v4().to_string();
    TempDir::new(&dir).expect("failed to create tmpdir")
}
