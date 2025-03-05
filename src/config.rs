use crate::preproc::Normalizer;
use crate::tokenizer::VocabMap;
use serde::{Deserialize, Serialize};


#[derive(Serialize, Deserialize, Clone)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub special_tokens_map: Option<VocabMap>,
    #[serde(default)]
    pub preproc: Normalizer,
}

impl TokenizerConfig {
    pub fn new(vocab_size: usize, preproc: Option<Normalizer>) -> Self {
        assert!(vocab_size > 0, "can't train on vocab_size <= 0!");

        let preproc = preproc.unwrap_or_default();

        Self {
            vocab_size,
            preproc,
            special_tokens_map: None,
        }
    }
}
