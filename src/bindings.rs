use pyo3::prelude::*;

//TODO: write macro for default wrapper impl ?

#[pymodule]
#[pyo3(name = "tkn")]
mod tkn {
    use super::*;
    use crate::config::TokenizerConfig as _TokenizerConfig;
    use crate::preproc::Normalizer as _Normalizer;
    use crate::BPETokenizer as _BPETokenizer;

    // TODO: can this become an enum?
    #[pyclass]
    #[derive(Clone)]
    struct Normalizer(_Normalizer);

    #[pyclass]
    #[derive(Clone)]
    struct TokenizerConfig(_TokenizerConfig);

    #[pymethods]
    impl TokenizerConfig {
        #[new]
        #[pyo3(signature=(vocab_size, preproc=None, /))]
        fn new(vocab_size: usize, preproc: Option<Normalizer>) -> Self {
            TokenizerConfig(_TokenizerConfig::new(
                vocab_size,
                preproc.map(|x| x.0),
            ))
        }
    }

    #[pyclass]
    struct BPETokenizer(_BPETokenizer);

    #[pymethods]
    impl BPETokenizer {
        #[new]
        fn new(config: TokenizerConfig) -> Self {
            BPETokenizer(_BPETokenizer::new(config.0))
        }

        #[pyo3(signature= (text="".to_string()))]
        fn preprocess(&self, mut text: String) -> String {
            self.0.preprocess(&mut text);
            text
        }
    }
}
