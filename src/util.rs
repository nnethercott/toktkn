use std::{
    collections::hash_map::{HashMap, RandomState},
    hash::BuildHasher,
};

/// replace `ngram`s in `tokens` with `replace`
pub fn ngram_replace<T>(tokens: &mut Vec<T>, ngram: &[T], replace: &[T])
where
    T: PartialEq + Copy,
{
    let mut remove = Vec::new();
    let mut prev_stack: usize = 0;

    tokens
        .windows(ngram.len())
        .enumerate()
        .for_each(|(i, win)| {
            if win == ngram && prev_stack == 0 {
                remove.push(i);
                prev_stack = ngram.len() - 1;
            } else {
                // so no match can occur in overlapping window
                prev_stack = match prev_stack.checked_sub(1) {
                    Some(num) => num,
                    _ => 0,
                };
            }
        });

    // NOTE: from a leetcode (can't remember which)
    for i in remove.iter().rev() {
        *tokens = [&tokens[0..*i], replace, &tokens[*i + ngram.len()..]].concat();
    }
}

pub fn replace_special_tokens<T, S>(tokens: &mut Vec<T>, map: &HashMap<String, T, S>)
where
    T: PartialEq + Copy,
    T: From<u8>,
    S: BuildHasher,
{
    // assume special tokens appear in no subwords
    map.iter().for_each(|(word, &t)|{
        let ngram: Vec<T> = word.chars().map(|c| T::from(c as u8)).collect();
        ngram_replace(tokens, &ngram, &[t]);
    });
}

pub fn inject_special_tokens<T, S>(tokens: &mut Vec<T>, map: &HashMap<String, T, S>)
where
    T: PartialEq + Copy,
    T: From<u8>,
    S: BuildHasher,
{
    // assume special tokens appear in no subwords
    map.iter().for_each(|(word, &t)|{
        let ngram: Vec<T> = word.chars().map(|c| T::from(c as u8)).collect();
        ngram_replace(tokens, &[t], &ngram);
    });
}
