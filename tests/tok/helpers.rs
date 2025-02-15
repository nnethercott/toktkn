use fake::faker::lorem::en::{Paragraph, Sentence};
use fake::{Fake, Faker};

pub fn get_corpus() -> String {
    let corpus: String = Paragraph(100..101).fake();
    corpus
}

pub fn get_sentence() -> String {
    let sentence: String = Sentence(10..50).fake();
    sentence
}
