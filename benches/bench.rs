#![feature(test)]

// Nice blog post on the subject :
// https://seenaburns.com/benchmarking-rust-with-cargo-bench/

use toktokenizer::{config::*, BPETokenizer, Tokenizer};

extern crate test;

#[cfg(test)]
mod tests {
    use super::*;
    use fake::faker::lorem::en::{Paragraph, Sentence};
    use fake::{Fake, Faker};
    use test::Bencher;

    fn get_corpus() -> String {
        let corpus: String = Paragraph(100..101).fake();
        corpus
    }

    // TODO: make this exactly N bytes
    fn get_sentence() -> String {
        let sentence: String = Sentence(10..50).fake();
        sentence
    }

    fn get_tokenizer() -> BPETokenizer {
        let config = TokenizerConfig::new(100, None);
        BPETokenizer::new(config)
    }

    // NOTE: bench is for micro-benchmarking, not heavy computation
    #[bench]
    fn bench_tokenizer(b: &mut Bencher) {
        let mut tokenizer = get_tokenizer();
        let corpus = get_corpus();
        tokenizer.train(&corpus);

        // test input
        let sentence = get_sentence();

        b.iter(|| tokenizer.encode(&sentence));
    }
}
