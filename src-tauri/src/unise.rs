//! UniSE speech enhancement client — stub for LLM hotkey branch.
//! ponytail: returns input unchanged. Real UniSE lives on feat/unise.
pub async fn enhance(wav_bytes: &[u8]) -> Vec<u8> {
    wav_bytes.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enhance_passthrough() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let input = b"test wav data".to_vec();
        let result = rt.block_on(enhance(&input));
        assert_eq!(result, input);
    }
}
