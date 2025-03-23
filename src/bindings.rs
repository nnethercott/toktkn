use pyo3::prelude::*;

#[pymodule]
#[pyo3(name = "toktkn")]
mod tkn {
    use pyo3::types::PyType;

    use super::*;
    use crate::{Token, Tokenizer, Pretrained, FwdMap};

    use crate::config::TokenizerConfig as _TokenizerConfig;
    use crate::preproc::Normalizer as _Normalizer;
    use crate::BPETokenizer as _BPETokenizer;

    #[pyclass]
    #[derive(Clone)]
    struct Normalizer(_Normalizer);

    #[pyclass]
    #[derive(Clone)]
    struct TokenizerConfig(_TokenizerConfig);

    #[pymethods]
    impl TokenizerConfig {
        #[new]
        #[pyo3(signature=(vocab_size, preproc=None, /))] // '/' each param before is positional-only
        fn new(vocab_size: usize, preproc: Option<Normalizer>) -> Self {
            TokenizerConfig(_TokenizerConfig::new(
                vocab_size,
                preproc.map(|x| x.0),
            ))
        }
        #[classmethod]
        pub fn from_pretrained(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
            let config = _TokenizerConfig::from_pretrained(path)?;
            Ok(TokenizerConfig(config))
        }

        pub fn save_pretrained(&mut self, path: &str) -> PyResult<()>{
            Ok(self.0.save_pretrained(path)?)
        }
    }

    #[pyclass]
    struct BPETokenizer(_BPETokenizer);

    #[pymethods]
    impl BPETokenizer {
        #[new]
        pub fn new(config: TokenizerConfig) -> Self {
            BPETokenizer(_BPETokenizer::new(config.0))
        }

        pub fn __len__(&self) -> usize{
            self.0.len()
        }

        #[getter]
        pub fn encoder(&self)->PyResult<FwdMap>{
            Ok(self.0.encoder.clone())
        }

        #[pyo3(signature= (text="".to_string()))]
        pub fn preprocess(&self, mut text: String) -> String {
            self.0.preprocess(&mut text);
            text
        }

        pub fn train(&mut self, text: &str) -> Vec<Token>{
            self.0.train(text)
        }

        pub fn encode(&mut self, text: &str) -> Vec<Token>{
            self.0.encode(text)
        }

        pub fn decode(&mut self, ids: Vec<Token>) -> String{
            self.0.decode(&ids)
        }

        pub fn add_special_tokens(&mut self, special_tokens: Vec<String>){
            self.0.add_special_tokens(special_tokens);
        }

        #[classmethod]
        pub fn from_pretrained(_cls: &Bound<'_, PyType>, path: &str) -> PyResult<Self> {
            let bpe = _BPETokenizer::from_pretrained(path)?;
            Ok(BPETokenizer(bpe))
        }

        pub fn save_pretrained(&mut self, path: &str) -> PyResult<()>{
            Ok(self.0.save_pretrained(path)?)
        }
    }
}
