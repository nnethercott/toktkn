use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHasher};
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_with::{serde_as, DisplayFromStr};
use std::error::Error;
use std::io::Write;
use std::sync::RwLock;
use std::{fs, str};

use crate::config::TokenizerConfig;
use crate::preproc::Normalize;
use crate::util::{inject_special_tokens, ngram_replace, replace_special_tokens};

use crate::pretrained::Pretrained;

pub type Token = u32; // 2^32 - 1 max new tokens

// map aliases
pub type FwdMap = FxHashMap<(Token, Token), Token>;
pub type BkwdMap = FxHashMap<Token, (Token, Token)>;
pub type VocabMap = FxHashMap<String, Token>;

pub trait Tokenizer {
    fn encode(&self, text: &str) -> Vec<Token>;
    fn decode(&self, input_ids: &[Token]) -> String;
}

#[serde_as]
#[derive(Serialize, Deserialize)]
pub struct BPETokenizer {
    #[serde_as(as = "Vec<((DisplayFromStr, DisplayFromStr), DisplayFromStr)>")]
    pub encoder: FwdMap,
    #[serde(skip)]
    pub decoder: RwLock<Option<BkwdMap>>, // thread-safe & nullable
    pub config: TokenizerConfig,
}

impl Tokenizer for BPETokenizer {
    fn encode(&self, text: &str) -> Vec<Token> {
        // parallel
        let first_pass: Vec<Token> = text
            .as_bytes()
            .par_chunks(1000)
            .flat_map_iter(|c| {
                let chunk_as_tokens: Vec<Token> = c.iter().map(|&i| i as Token).collect();
                self._encode_chunk(&chunk_as_tokens)
            })
            .collect();

        // sequential cleanup
        self._encode_chunk(&first_pass)
    }

    fn decode(&self, input_ids: &[Token]) -> String {
        // first pass
        let first_pass: Vec<Token> = input_ids
            .par_chunks(1000)
            .flat_map_iter(|t| self._decode_chunk(t))
            .collect();

        //second pass
        let utf8 = self
            ._decode_chunk(&first_pass)
            .iter()
            .map(|&i| i as u8)
            .collect::<Vec<u8>>();

        let s =
            str::from_utf8(&utf8).expect(&format!("failed to decode into valid utf-8: {:?}", utf8));
        s.into()
    }
}

impl BPETokenizer {
    pub fn new(config: TokenizerConfig) -> Self {
        Self {
            encoder: FwdMap::default(),
            decoder: RwLock::new(None),
            config,
        }
    }

    pub fn len(&self) -> usize {
        match self.config.special_tokens_map.as_ref() {
            Some(map) => map.len() + self.encoder.len(),
            None => self.encoder.len(),
        }
    }

    fn _sync_decoder(&self) {
        let mut inner = self.decoder.write().unwrap();
        inner.replace(
            self.encoder
                .iter()
                .map(|(&k, &v)| (v, k))
                .collect::<BkwdMap>(),
        );
    }

    pub fn add_special_tokens<S: Into<String>>(&mut self, tokens: Vec<S>) {
        let token_id = self.len() + 128;
        let token_map: VocabMap = tokens
            .into_iter()
            .enumerate()
            .map(|(e, s)| (s.into(), (token_id + e) as Token))
            .collect();

        self.config.special_tokens_map =
            self.config
                .special_tokens_map
                .take()
                .map_or(Some(token_map.clone()), |mut m| {
                    m.extend(token_map.into_iter());
                    Some(m)
                });
    }

    pub fn preprocess(&self, text: &mut String) {
        let preproc = self.config.preproc.into_strategy();
        preproc.normalize(text);
    }

    fn _encode_chunk(&self, chunk: &[Token]) -> Vec<Token> {
        let mut tokens = chunk.to_vec().clone();

        if let Some(map) = self.config.special_tokens_map.as_ref() {
            replace_special_tokens::<Token, FxBuildHasher>(&mut tokens, map);
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

    // TODO: custom errors ?
    fn _decode_chunk(&self, tokens: &[Token]) -> Vec<Token> {
        let mut tokens: Vec<Token> = Vec::from(tokens);

        // lazy init
        self._sync_decoder();
        let lock = self.decoder.read().expect("could not acquire lock");

        let decoder = lock.as_ref().unwrap();

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

        // special tokens
        if let Some(map) = self.config.special_tokens_map.as_ref() {
            inject_special_tokens::<Token, FxBuildHasher>(&mut tokens, map);
        }

        tokens
    }

    // TODO: optimize with multi processing and broadcast ?
    pub fn train(&mut self, text: &str) -> Vec<Token> {
        let mut pieces: Vec<Token>;

        if !self.encoder.is_empty() {
            println!("pretrained tokenizer detected!");
            pieces = self.encode(text);
        } else {
            let text = text.as_bytes();
            pieces = text.iter().map(|&i| i as Token).collect();
        }

        match self.config.vocab_size.checked_sub(self.len()) {
            Some(size) => {
                for _ in tqdm::tqdm(0..size) {
                    let mut counts = FwdMap::default();
                    for i in 0..pieces.len() - 1 {
                        *counts.entry((pieces[i], pieces[i + 1])).or_insert(0) += 1;
                    }

                    let (&p, _) = counts.iter().max_by_key(|(_, &c)| c).unwrap();
                    let token_id = (self.len() + 127 + 1) as Token;

                    self.encoder.insert(p, token_id);
                    ngram_replace(&mut pieces, &[p.0, p.1], &[token_id]);
                }
            }
            None => println!(
                "requested vocab_size: {} already reached.",
                self.config.vocab_size
            ),
        };

        self._sync_decoder();
        pieces
    }
}
