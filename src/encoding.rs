//! Build independent token rows for Kev's hybrid Qwen backbones.
//!
//! Tokenization and safe escaping of user text are the caller's responsibility.
//! This module rejects control-token IDs in user-provided content so callers
//! cannot accidentally introduce forged question or option delimiters.

use std::error::Error;
use std::fmt;

/// Token IDs of the checkpoint's five delimiter tokens.
#[derive(Debug, Clone, Copy)]
pub struct Delimiters {
    pub state: u32,
    pub question: u32,
    pub option_start: u32,
    pub option_end: u32,
    pub decide: u32,
}

/// One tokenized instruction and its tokenized options, in response order.
#[derive(Debug, Clone)]
pub struct Question<'a> {
    pub instruction: &'a [u32],
    pub options: &'a [Vec<u32>],
}

/// A shared prefix and independent causal question rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncodedRequest {
    pub prefix: Vec<u32>,
    pub rows: Vec<QuestionRow>,
}

/// A branch continuing from the prefix, with indices relative to this row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestionRow {
    pub token_ids: Vec<u32>,
    pub position_ids: Vec<usize>,
    pub decision_index: usize,
    pub option_end_indices: Vec<usize>,
}

/// Encode independent question rows using strict, non-truncating limits.
///
/// `max_state_tokens` includes the state delimiter; `max_row_tokens` includes
/// the prefix and the row. A caller can prefill `prefix` once and evaluate
/// each row with its own copy of that prefix state. Qwen3.5's recurrent
/// layers cannot safely be given a packed block-causal attention mask in place
/// of independent rows.
pub fn encode_rows(
    state: &[u32],
    questions: &[Question<'_>],
    delimiters: Delimiters,
    max_state_tokens: usize,
    max_row_tokens: usize,
) -> Result<EncodedRequest, EncodingError> {
    let controls = [
        delimiters.state,
        delimiters.question,
        delimiters.option_start,
        delimiters.option_end,
        delimiters.decide,
    ];
    for (index, value) in controls.iter().enumerate() {
        if controls[..index].contains(value) {
            return Err(EncodingError::DuplicateDelimiters);
        }
    }
    let no_control = |tokens: &[u32]| {
        if tokens.iter().any(|token| controls.contains(token)) {
            Err(EncodingError::ControlTokenInContent)
        } else {
            Ok(())
        }
    };
    no_control(state)?;
    let prefix_len = state.len().checked_add(1).ok_or(EncodingError::TooLong)?;
    if prefix_len > max_state_tokens || prefix_len > max_row_tokens {
        return Err(EncodingError::TooLong);
    }
    let mut prefix = Vec::with_capacity(prefix_len);
    prefix.push(delimiters.state);
    prefix.extend_from_slice(state);

    let mut rows = Vec::with_capacity(questions.len());
    for question in questions {
        if !(1..=255).contains(&question.options.len()) {
            return Err(EncodingError::InvalidOptionCount);
        }
        no_control(question.instruction)?;
        let mut branch_len = question
            .instruction
            .len()
            .checked_add(2) // question + decide
            .ok_or(EncodingError::TooLong)?;
        for option in question.options {
            no_control(option)?;
            branch_len = branch_len
                .checked_add(option.len())
                .and_then(|len| len.checked_add(2)) // option delimiters
                .ok_or(EncodingError::TooLong)?;
        }
        let total = prefix_len
            .checked_add(branch_len)
            .ok_or(EncodingError::TooLong)?;
        if total > max_row_tokens {
            return Err(EncodingError::TooLong);
        }
        let mut token_ids = Vec::with_capacity(branch_len);
        let mut option_end_indices = Vec::with_capacity(question.options.len());
        token_ids.push(delimiters.question);
        token_ids.extend_from_slice(question.instruction);
        for option in question.options {
            token_ids.push(delimiters.option_start);
            token_ids.extend_from_slice(option);
            option_end_indices.push(token_ids.len());
            token_ids.push(delimiters.option_end);
        }
        let decision_index = token_ids.len();
        token_ids.push(delimiters.decide);
        rows.push(QuestionRow {
            position_ids: (prefix_len..total).collect(),
            token_ids,
            decision_index,
            option_end_indices,
        });
    }
    Ok(EncodedRequest { prefix, rows })
}

/// A request that cannot be encoded safely or within its limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingError {
    DuplicateDelimiters,
    ControlTokenInContent,
    InvalidOptionCount,
    TooLong,
}

impl fmt::Display for EncodingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::DuplicateDelimiters => "delimiter token IDs must be distinct",
            Self::ControlTokenInContent => "user content contains a delimiter token ID",
            Self::InvalidOptionCount => "each question must have between 1 and 255 options",
            Self::TooLong => "encoded request exceeds its token limit",
        })
    }
}

impl Error for EncodingError {}
