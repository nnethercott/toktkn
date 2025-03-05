use crate::config::TokenizerConfig;
use crate::FwdMap;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr};
use std::fs::{read_to_string, File};
use std::path::Path;

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

#[serde_as]
#[derive(Serialize, Deserialize)]
struct HashMapWrapper{ 
    #[serde_as(as = "Vec<((DisplayFromStr, DisplayFromStr), DisplayFromStr)>")]
    encoder: FwdMap
}

#[derive(Serialize, Deserialize)]
struct BPETokenizerWrapper {
    config: TokenizerConfig,
    encoder: HashMapWrapper,
}

// impl Pretrained for BPETokenizer {
//     fn save_pretrained<P: AsRef<std::path::Path>>(&self, path: P) -> Result<(), std::io::Error> {
//         let wrapper = BPETokenizerWrapper {
//             config: self.config.clone(),
//             encoder: HashMapWrapper{encoder: self.encoder.clone()}
//         };
//         wrapper.save_pretrained(path)
//     }
//
//     fn from_pretrained<P: AsRef<std::path::Path>>(path: P) -> Result<Self, std::io::Error> {
//         let wrapper = BPETokenizerWrapper::from_pretrained(path)?;
//         let mut tokenizer = BPETokenizer::new(wrapper.config);
//         tokenizer.encoder = wrapper.encoder.encoder;
//         Ok(tokenizer)
//     }
// }
