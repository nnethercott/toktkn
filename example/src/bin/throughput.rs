use std::time::Instant;
use toktokenizer::*;

const TEXT: &'static str = include_str!("../../../benches/corpus.txt");

fn time<F, T>(f: F) -> (T, u128)
where
    F: FnOnce() -> T,
{
    let now = Instant::now();
    let res = f();
    let elapsed = now.elapsed();
    
    (res, elapsed.as_millis())
}

fn main() {
    let tok = BPETokenizer::from_pretrained("tokenizer.json").expect("generate tokenizer.json first!");
    let text = TEXT.repeat(300);
    let mb = (text.chars().count() as f32)*(8f32 / 1e6_f32);

    let f = ||{
        tok.encode(&text)
    };

    let (tokens, delta) = time(f);

    println!("tokenized {} mb in {:?} ms", mb, delta);
    println!("throughput: {} mb/s", 1000f32*mb/(delta as f32));
    println!("compression ratio: {}", (tokens.len() as f32)/(text.chars().count() as f32));
}
