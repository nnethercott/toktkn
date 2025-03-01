use crate::preproc::{DefaultNormalizer, Normalize, Normalizer};
use crate::tokenizer::VocabMap;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::{read_to_string, File};
use std::io::{Read, Write};
use std::path::Path;

use crate::pretrained::Pretrained;

#[derive(Serialize, Deserialize, Clone)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub special_tokens_map: Option<VocabMap>,
    #[serde(default)]
    pub preproc: Normalizer,
}

//TODO: more methods to add special tokens ?

impl TokenizerConfig {
    pub fn new(vocab_size: usize, preproc: Option<Normalizer>) -> Self {
        assert!(vocab_size > 0, "can't train on vocab_size <= 0!");

        let preproc = match preproc {
            Some(p) => p,
            None => Normalizer::default(),
        };

        Self {
            vocab_size,
            preproc,
            special_tokens_map: None,
        }
    }
}
