use crate::preproc::{DefaultNormalizer, Normalize};
use crate::tokenizer::VocabMap;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::{read_to_string, File};
use std::io::{Read, Write};

// NOTE: some trait `Pretrainable<T: Serialize>` with default impl for read/write from
// disk !
// would also have new file `traits.rs` maybe

// NOTE: tokenizer save_pretrained should convert its own token map into tokenizer config
// before serializing

pub trait Pretrained : Serialize +  for<'a> Deserialize<'a> {

    fn save_pretrained(&self, path: &str) -> Result<(), std::io::Error> {
        let file = File::create(path)?;
        serde_json::to_writer(file, &self).expect("failed to save pretrained !");
        Ok(())
    }

    fn from_pretrained(path: &str) -> Result<Self, std::io::Error> {
        let s = read_to_string(path)?;
        let config =
            serde_json::from_str::<Self>(&s).expect("failed to load pretrained");
        Ok(config)
    }
}

#[derive(Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub special_tokens_map: Option<VocabMap>,
    // pub preproc: Box<dyn Normalize>,
}

impl TokenizerConfig {
    pub fn new(vocab_size: usize) -> Self {
        assert!(vocab_size > 0, "can't train on vocab_size <= 0!");

        Self {
            vocab_size,
            special_tokens_map: None,
            // preproc: Box::new(DefaultNormalizer),
        }
    }
}

impl Pretrained for TokenizerConfig {
}

#[cfg(test)]
mod tests{
    use super::*;

    #[test]
    fn test_serialize(){
        //serialize to tmp file (maybe need crate)
        //passes
    }
}
