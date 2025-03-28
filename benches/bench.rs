#![feature(test)]
// Nice blog post on the subject :
// https://seenaburns.com/benchmarking-rust-with-cargo-bench/

use toktkn::{config::*, BPETokenizer, Tokenizer};

extern crate test;

const CORPUS: &'static str =  include_str!("corpus.txt");
const SAMPLE: &'static str =  include_str!("sample.txt");

fn get_tokenizer() -> BPETokenizer {
    let config = TokenizerConfig::new(100, None);
    let mut tok = BPETokenizer::new(config);
    tok.train(CORPUS);
    tok
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[bench]
    fn bench_tokenizer(b: &mut Bencher) {
        let tokenizer = get_tokenizer();

        b.iter(|| tokenizer.encode(SAMPLE));
    }
}
