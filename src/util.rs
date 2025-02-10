use rustc_hash::FxHashMap;

/// merge step logic. replaces bigram `find` new symbol `replace`
pub fn byte_pair_merge<T>(pieces: &mut Vec<T>, find: (T, T), replace: T)
where
    T: PartialEq + Copy,
{
    let mut remove = Vec::new();
    let mut prev: bool = true;

    pieces.windows(2).enumerate().for_each(|(i, x)| {
        if (x[0], x[1]) == find && prev {
            remove.push(i + 1);
            prev = false;
        } else {
            // so no match can occur in overlapping window
            prev = true;
        }
    });

    // NOTE: from a leetcode (can't remember which)
    for i in remove.iter().rev() {
        pieces[i - 1] = replace;
        pieces.remove(*i);
    }
}

