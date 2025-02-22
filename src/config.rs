use crate::preproc::{DefaultNormalizer, Normalize, Normalizer};
use crate::tokenizer::VocabMap;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fs::{read_to_string, File};
use std::io::{Read, Write};
use std::path::Path;

// NOTE: impl `Pretrained` for tokenizer?

pub trait Pretrained: Sized {
    fn save_pretrained<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error>;
    fn from_pretrained<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error>;
}

impl<T> Pretrained for T
where
    T: Serialize + for<'a> Deserialize<'a>,
{
    fn save_pretrained<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let file = File::create(path)?;
        serde_json::to_writer(file, &self).expect("failed to save pretrained !");
        Ok(())
    }

    fn from_pretrained<P: AsRef<Path>>(path: P) -> Result<Self, std::io::Error> {
        let s = read_to_string(path)?;
        let config = serde_json::from_str::<Self>(&s).expect("failed to load pretrained");
        Ok(config)
    }
}

#[derive(Serialize, Deserialize)]
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

#[cfg(test)]
mod tests {
    use super::*;
    extern crate tempdir;
    use tempdir::TempDir;

    #[test]
    fn test_serialize() -> std::io::Result<()> {
        // arrange
        let dir = TempDir::new("tmp_dir_for_configs")?;
        let file_path = dir.path().join("config.txt");
        let mut config = TokenizerConfig::new(42, None);

        // act
        config.save_pretrained(&file_path)?;
        assert!(file_path.exists());

        config = TokenizerConfig::from_pretrained(&file_path)?;
        assert_eq!(config.vocab_size, 42);
        assert_eq!(config.preproc, Normalizer::default());

        dir.close()?;

        Ok(())
    }
}
