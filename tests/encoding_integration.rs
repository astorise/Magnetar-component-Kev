use magnetar_component_kev::encoding::{Delimiters, EncodingError, Question, encode_rows};
use serde::Deserialize;

#[derive(Deserialize)]
struct UpstreamRows {
    prefix: Vec<u32>,
    prefix_positions: Vec<usize>,
    rows: Vec<UpstreamRow>,
}

#[derive(Deserialize)]
struct UpstreamRow {
    ids: Vec<u32>,
    pos: Vec<usize>,
    decide: usize,
    opts: Vec<usize>,
}

fn delimiters() -> Delimiters {
    Delimiters {
        state: 1000,
        question: 1001,
        option_start: 1002,
        option_end: 1003,
        decide: 1004,
    }
}

#[test]
fn hybrid_question_rows_match_the_upstream_encoder_and_rows_of() {
    // Generated independently with upstream kev/model.py at
    // 65b1e763943ace53fd3a4aae00d8d08dbd849772, using a tokenizer whose
    // text token IDs are ASCII and delimiter IDs are 1000..1004.
    let first_options = vec![vec![68], vec![69, 70]];
    let second_options = vec![vec![72]];
    let questions = [
        Question {
            instruction: &[67],
            options: &first_options,
        },
        Question {
            instruction: &[71],
            options: &second_options,
        },
    ];
    let encoded = encode_rows(&[65, 66], &questions, delimiters(), 3, 13).unwrap();
    let expected: UpstreamRows =
        serde_json::from_slice(include_bytes!("fixtures/kev-upstream-rows.json")).unwrap();
    assert_eq!(encoded.prefix, expected.prefix);
    assert_eq!(
        (0..encoded.prefix.len()).collect::<Vec<_>>(),
        expected.prefix_positions
    );
    assert_eq!(encoded.rows.len(), expected.rows.len());
    for (row, expected) in encoded.rows.iter().zip(&expected.rows) {
        assert_eq!(row.token_ids, expected.ids);
        assert_eq!(row.position_ids, expected.pos);
        assert_eq!(row.option_end_indices, expected.opts);
        assert_eq!(row.decision_index, expected.decide);
    }
}

#[test]
fn strict_limits_include_the_prefix_and_never_truncate() {
    let options = vec![vec![68], vec![69, 70]];
    let question = Question {
        instruction: &[67],
        options: &options,
    };
    assert_eq!(
        encode_rows(
            &[65, 66],
            std::slice::from_ref(&question),
            delimiters(),
            2,
            13
        ),
        Err(EncodingError::TooLong)
    );
    assert_eq!(
        encode_rows(&[65, 66], &[question], delimiters(), 3, 12),
        Err(EncodingError::TooLong)
    );
}

#[test]
fn rejects_control_token_injection_and_invalid_option_counts() {
    let options = vec![vec![68]];
    let question = Question {
        instruction: &[67],
        options: &options,
    };
    assert_eq!(
        encode_rows(
            &[1002],
            std::slice::from_ref(&question),
            delimiters(),
            20,
            40
        ),
        Err(EncodingError::ControlTokenInContent)
    );
    let forged_options = vec![vec![1004]];
    let forged = Question {
        instruction: &[],
        options: &forged_options,
    };
    assert_eq!(
        encode_rows(&[], &[forged], delimiters(), 20, 40),
        Err(EncodingError::ControlTokenInContent)
    );
    let duplicate = Delimiters {
        decide: 1000,
        ..delimiters()
    };
    assert_eq!(
        encode_rows(&[], &[question], duplicate, 20, 40),
        Err(EncodingError::DuplicateDelimiters)
    );
    let many_options = vec![Vec::new(); 256];
    let too_many = Question {
        instruction: &[],
        options: &many_options,
    };
    assert_eq!(
        encode_rows(&[], &[too_many], delimiters(), 20, 1000),
        Err(EncodingError::InvalidOptionCount)
    );
    let empty = Question {
        instruction: &[],
        options: &[],
    };
    assert_eq!(
        encode_rows(&[], &[empty], delimiters(), 20, 40),
        Err(EncodingError::InvalidOptionCount)
    );
}
