use toktokenizer::{
    preproc::{DefaultNormalizer, Normalize},
    Tokenizer, BPETokenizer
};

use fake::faker::lorem::en::{Paragraph, Sentence};
use fake::{Fake, Faker};

//TODO: download wikitext

fn main() {
    let normalizer = DefaultNormalizer;
    let mut tok = BPETokenizer::new(&normalizer);

    let corpus: String = Paragraph(100..101).fake();

    tok.train(&corpus, 50);
    println!("{}", tok.len());

    tok.train(&corpus, 75);
    println!("{}", tok.len());
}
