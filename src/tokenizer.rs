use rustc_hash::{FxBuildHasher, FxHashMap, FxHasher};
use std::cell::RefCell;
use std::error::Error;
use std::io::Write;
use std::{fs, str};

use crate::preproc::{Normalize, DEFAULT_NORMALIZER};
use crate::util::{apply_special_tokens_map, byte_pair_merge};

pub type Token = u32; // 2^32 - 1 max new tokens

// map aliases
pub type FwdMap = FxHashMap<(Token, Token), Token>;
pub type BkwdMap = FxHashMap<Token, (Token, Token)>;
pub type VocabMap = FxHashMap<String, Token>;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<Token>;
    fn decode(&self, input_ids: &[Token]) -> String;
}

// TODO: add tokenizer config

pub struct BPETokenizer<'a> {
    pub normalizer: &'a dyn Normalize,
    pub encoder: FwdMap,
    pub special_tokens_map: Option<VocabMap>,
    pub decoder: RefCell<Option<BkwdMap>>,
}

impl<'a> Tokenizer for BPETokenizer<'_> {
    fn encode(&self, text: &str) -> Vec<Token> {
        let text = text.as_bytes();
        self._encode_chunk(text)
    }

    fn decode(&self, input_ids: &[Token]) -> String {
        let utf8: Vec<u8> = self._decode_chunk(input_ids);
        let s = str::from_utf8(&utf8).expect(&format!("failed to decode into valid utf-8: {:?}", utf8));
        s.into()
    }
}

impl<'a> BPETokenizer<'a> {
    pub fn new(normalizer: &'a dyn Normalize) -> Self {
        Self {
            normalizer,
            encoder: FwdMap::default(),
            special_tokens_map: None,
            decoder: RefCell::new(None),
        }
    }

    pub fn len(&self) -> usize {
        match self.special_tokens_map.as_ref() {
            Some(map) => map.len() + self.encoder.len(),
            None => self.encoder.len(),
        }
    }

    fn _sync_decoder(&self) {
        let _ = self.decoder.replace(Some(
            self.encoder
                .iter()
                .map(|(&k, &v)| (v, k))
                .collect::<BkwdMap>(),
        ));
    }

    fn add_special_tokens<S: Into<String>>(&mut self, tokens: Vec<S>) {
        let token_id = self.len() + 128;
        let token_map: VocabMap = tokens
            .into_iter()
            .enumerate()
            .map(|(e, s)| (s.into(), (token_id + e) as Token))
            .collect();

        self.special_tokens_map =
            self.special_tokens_map
                .take()
                .map_or(Some(token_map.clone()), |mut m| {
                    m.extend(token_map.into_iter());
                    Some(m)
                });
    }

    pub fn from_pretrained(file: &str) -> Result<Self, Box<dyn Error>> {
        let mut tok = Self::new(&DEFAULT_NORMALIZER);
        tok.load_encoder(file)?;
        Ok(tok)
    }

    pub fn preprocess(&self, text: &mut String) {
        self.normalizer.normalize(text);
    }

    pub fn save_pretrained(&self, file: &str) -> Result<(), Box<dyn std::error::Error>> {
        // need to reverse key-value order since serde can't serialize tuples as map keys
        self._sync_decoder();
        let binding = self.decoder.borrow();
        let decoder = binding.as_ref().unwrap();

        let serialized = serde_json::to_string(decoder)?;
        let mut f = fs::File::create(file)?;

        f.write_all(serialized.as_bytes())?;

        Ok(())
    }

    pub fn load_encoder(&mut self, file: &str) -> Result<(), Box<dyn std::error::Error>> {
        let encoder_str = fs::read_to_string(file)?;
        let _encoder: FxHashMap<Token, (Token, Token)> = serde_json::from_str(&encoder_str)?;
        let encoder: FwdMap = _encoder.iter().map(|(&k, &v)| (v, k)).collect();

        self.encoder = encoder;
        self._sync_decoder();

        Ok(())
    }

    fn _encode_chunk(&self, chunk: &[u8]) -> Vec<Token> {
        let mut tokens: Vec<Token> = chunk.to_vec().iter().map(|&x| x as Token).collect();
        if let Some(map) = self.special_tokens_map.as_ref() {
            apply_special_tokens_map::<Token, FxBuildHasher>(&mut tokens, map);
        }

        loop {
            let mut merges = Vec::new();

            for i in 0..tokens.len() - 1 {
                if let Some(&new_token) = self.encoder.get(&(tokens[i], tokens[i + 1])) {
                    merges.push((i, new_token));
                }
            }
            // early stopping: no more token pairs in merge rules
            if merges.is_empty() {
                break;
            }

            // apply merges and swap in tokens in reverse
            let mut i = merges.len() - 1;

            while i > 0 {
                let x = &mut merges[i - 1..=i];
                let mut l = x[0];
                let r = x[1];

                if r.0 - l.0 > 1 && r.1 != Token::MAX {
                    tokens[r.0] = r.1;
                    tokens.remove(r.0 + 1);
                } else if r.1 < l.1 {
                    tokens[r.0] = r.1;
                    tokens.remove(r.0 + 1);

                    l.1 = Token::MAX;
                    i -= 1;
                }

                //avoid overflow on usize 0-1
                if i == 0 {
                    break;
                }
                i -= 1;
            }

            // edge case
            if merges.len() == 1 || merges[0].1 < merges[1].1 {
                tokens[merges[0].0] = merges[0].1;
                tokens.remove(merges[0].0 + 1);
            }
        }
        tokens
    }

    fn _decode_chunk(&self, chunk: &[Token]) -> Vec<u8> {
        let mut tokens: Vec<Token> = Vec::from(chunk);

        //FIXME: direct search and replace with special_tokens_map

        // lazy init
        self._sync_decoder();
        let binding = self.decoder.borrow();
        let decoder = binding.as_ref().unwrap();

        loop {
            let mut demerges = Vec::new();
            for i in 0..tokens.len() {
                let rank = tokens[i];
                if let Some(&tup) = decoder.get(&rank) {
                    demerges.push((i, tup));
                }
            }
            if demerges.is_empty() {
                break;
            }

            for op in demerges.iter().rev() {
                let i = op.0;
                let tup = op.1;
                tokens[i] = tup.0;
                tokens.insert(i + 1, tup.1);
            }
        }

        tokens.iter().map(|&x| x as u8).collect()
    }

    pub fn train(&mut self, text: &str, vocab_size: usize) -> Vec<Token> {
        assert!(vocab_size > 0, "can't train on vocab_size <= 0!");
        let mut pieces: Vec<Token>;

        if !self.encoder.is_empty() {
            println!("pretrained tokenizer detected!");
            pieces = self.encode(text);
        } else {
            let text = text.as_bytes();
            pieces = text.iter().map(|&i| i as Token).collect();
        }

        match vocab_size.checked_sub(self.len()) {
            Some(size) => {
                for _ in tqdm::tqdm(0..size) {
                    let mut counts = FwdMap::default();
                    for i in 0..pieces.len() - 1 {
                        *counts.entry((pieces[i], pieces[i + 1])).or_insert(0) += 1;
                    }

                    let (&p, _) = counts.iter().max_by_key(|(_, &c)| c).unwrap();
                    let token_id = (self.len() + 127 + 1) as Token;

                    self.encoder.insert(p, token_id);
                    byte_pair_merge(&mut pieces, p, token_id);
                }
            }
            None => println!("requested vocab_size: {} already reached.", vocab_size),
        };

        self._sync_decoder();
        pieces
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preproc::DEFAULT_NORMALIZER;
    use fake::faker::lorem::en::{Paragraph, Sentence};
    use fake::faker::lorem::raw::Paragraphs;
    use fake::{Fake, Faker};

    fn get_corpus() -> String {
        let corpus: String = Paragraph(100..101).fake();
        corpus
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
    fn test_special_tokens_doesnt_break_encoding(){
        let mut tok = BPETokenizer::new(&DEFAULT_NORMALIZER);
        let special_tokens = vec!["<s>", "hello", "world", "</s>"];
        tok.add_special_tokens(special_tokens);

        let corpus = get_corpus();
        tok.train(&corpus, 10);
        let special_tokens = vec!["<nate>"];


        let sample = "hello hello world <s></s> some more text goes here";
        assert_eq!(tok.decode(&tok.encode(sample)), sample);


        dbg!(tok.encode(sample));
    }
}
