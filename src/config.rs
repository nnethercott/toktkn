use crate::tokenizer::VocabMap;
use serde::{Deserialize, Serialize};

// NOTE: some trait `Pretrainable<T: Serialize>` with default impl for read/write from
// disk !
// would also have new file `traits.rs` maybe


// NOTE: tokenizer save_pretrained should convert its own token map into tokenizer config 
// before serializing

trait Pretrained{
    pub fn from_pretrained(path: &str)->Self{
        todo!()
    }
    pub fn save_pretrained(path: &str){
        todo!()
    }
}

impl Pretrained for TokenizerConfig{
    todo!();
}

#[derive(Serialize, Deserialize)]
pub struct TokenizerConfig {
    pub vocab_size: usize,
    pub special_tokens_map: Option<VocabMap>,
    // some field for normalization/preprocessing?
}

impl TokenizerConfig {
    pub fn new(vocab_size: usize) -> Self {
        assert!(vocab_size > 0, "can't train on vocab_size <= 0!");

        Self {
            vocab_size,
            special_tokens_map: None,
        }
    }
}

//TODO: serde stuff
// load from config
// parse from string ?
// `from_pretrained` --> packge level fn Tokenizer::from_pretrained(...)

// pub fn from_pretrained(file: &str) -> Result<Self, Box<dyn Error>> {
//     let mut tok = Self::new(&DEFAULT_NORMALIZER);
//     tok.load_encoder(file)?;
//     Ok(tok)
// }
//
//
// pub fn save_pretrained(&self, file: &str) -> Result<(), Box<dyn std::error::Error>> {
//     // need to reverse key-value order since serde can't serialize tuples as map keys
//     self._sync_decoder();
//     let binding = self.decoder.borrow();
//     let decoder = binding.as_ref().unwrap();
//
//     let serialized = serde_json::to_string(decoder)?;
//     let mut f = fs::File::create(file)?;
//
//     f.write_all(serialized.as_bytes())?;
//
//     Ok(())
// }
//
// pub fn load_encoder(&mut self, file: &str) -> Result<(), Box<dyn std::error::Error>> {
//     let encoder_str = fs::read_to_string(file)?;
//     let _encoder: FxHashMap<Token, (Token, Token)> = serde_json::from_str(&encoder_str)?;
//     let encoder: FwdMap = _encoder.iter().map(|(&k, &v)| (v, k)).collect();
//
//     self.encoder = encoder;
//     self._sync_decoder();
//
//     Ok(())
// }
