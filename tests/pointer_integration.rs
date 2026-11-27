use std::collections::BTreeMap;

use magnetar_component_kev::encoding::{Delimiters, Question, encode_rows};
use magnetar_component_kev::{CheckpointTensor, HeadError, LinearProjection, PointerHead};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    input_dim: usize,
    pointer_dim: usize,
    temperature: f32,
    query_weights: Vec<f32>,
    query_bias: Vec<f32>,
    key_weights: Vec<f32>,
    key_bias: Vec<f32>,
    questions: Vec<ReferenceQuestion>,
}

#[derive(Deserialize)]
struct ReferenceQuestion {
    decision: Vec<f32>,
    options: Vec<Vec<f32>>,
    logits: Vec<f32>,
    probabilities: Vec<f32>,
}

#[test]
fn published_kev_08b_head_matches_pytorch_for_sparse_hidden_states() {
    let reference: Reference =
        serde_json::from_slice(include_bytes!("fixtures/kev-0.8b-pointer.json")).unwrap();
    let head = PointerHead::from_checkpoint_tensors(
        BTreeMap::from([
            (
                "q.weight".into(),
                CheckpointTensor {
                    shape: vec![reference.pointer_dim, reference.input_dim],
                    values: reference.query_weights.clone(),
                },
            ),
            (
                "q.bias".into(),
                CheckpointTensor {
                    shape: vec![reference.pointer_dim],
                    values: reference.query_bias.clone(),
                },
            ),
            (
                "k.weight".into(),
                CheckpointTensor {
                    shape: vec![reference.pointer_dim, reference.input_dim],
                    values: reference.key_weights.clone(),
                },
            ),
            (
                "k.bias".into(),
                CheckpointTensor {
                    shape: vec![reference.pointer_dim],
                    values: reference.key_bias.clone(),
                },
            ),
        ]),
        reference.temperature,
    )
    .unwrap();
    let tokenized_options: Vec<Vec<Vec<u32>>> = reference
        .questions
        .iter()
        .map(|question| question.options.iter().map(|_| vec![5]).collect())
        .collect();
    let questions: Vec<_> = tokenized_options
        .iter()
        .map(|options| Question {
            instruction: &[4],
            options,
        })
        .collect();
    let encoded = encode_rows(
        &[1, 2],
        &questions,
        Delimiters {
            state: 100,
            question: 101,
            option_start: 102,
            option_end: 103,
            decide: 104,
        },
        16,
        128,
    )
    .unwrap();
    for (index, question) in reference.questions.iter().enumerate() {
        let options: Vec<&[f32]> = question.options.iter().map(Vec::as_slice).collect();
        let actual = head.score(&question.decision, &options).unwrap();
        let row = &encoded.rows[index];
        let mut states = vec![0.0; row.token_ids.len() * reference.input_dim];
        states[row.decision_index * reference.input_dim
            ..(row.decision_index + 1) * reference.input_dim]
            .copy_from_slice(&question.decision);
        for (&at, state) in row.option_end_indices.iter().zip(&question.options) {
            states[at * reference.input_dim..(at + 1) * reference.input_dim].copy_from_slice(state);
        }
        assert_eq!(head.score_row(row, &states).unwrap(), actual);
        assert_eq!(actual.logits.len(), question.logits.len());
        for (actual, expected) in actual.logits.iter().zip(&question.logits) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "question {index} logit {actual} != {expected}"
            );
        }
        for (actual, expected) in actual.probabilities.iter().zip(&question.probabilities) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "question {index} probability {actual} != {expected}"
            );
        }
    }
}

fn small_inventory() -> BTreeMap<String, CheckpointTensor> {
    BTreeMap::from([
        (
            "q.weight".into(),
            CheckpointTensor {
                shape: vec![2, 2],
                values: vec![1.0, 0.0, 0.0, 1.0],
            },
        ),
        (
            "q.bias".into(),
            CheckpointTensor {
                shape: vec![2],
                values: vec![0.0; 2],
            },
        ),
        (
            "k.weight".into(),
            CheckpointTensor {
                shape: vec![2, 2],
                values: vec![1.0, 0.0, 0.0, 1.0],
            },
        ),
        (
            "k.bias".into(),
            CheckpointTensor {
                shape: vec![2],
                values: vec![0.0; 2],
            },
        ),
    ])
}

#[test]
fn checkpoint_inventory_rejects_missing_extra_malformed_and_nonfinite_tensors() {
    let mut missing = small_inventory();
    missing.remove("q.bias");
    assert!(matches!(
        PointerHead::from_checkpoint_tensors(missing, 1.0),
        Err(HeadError::InvalidTensorInventory)
    ));
    let mut extra = small_inventory();
    extra.insert(
        "unexpected".into(),
        CheckpointTensor {
            shape: vec![],
            values: vec![],
        },
    );
    assert!(matches!(
        PointerHead::from_checkpoint_tensors(extra, 1.0),
        Err(HeadError::InvalidTensorInventory)
    ));
    let mut wrong_shape = small_inventory();
    wrong_shape.get_mut("k.weight").unwrap().shape = vec![1, 4];
    assert!(matches!(
        PointerHead::from_checkpoint_tensors(wrong_shape, 1.0),
        Err(HeadError::InvalidTensorInventory)
    ));
    let mut wrong_values = small_inventory();
    wrong_values.get_mut("q.bias").unwrap().values.pop();
    assert!(matches!(
        PointerHead::from_checkpoint_tensors(wrong_values, 1.0),
        Err(HeadError::InvalidDimensions)
    ));
    let mut nonfinite = small_inventory();
    nonfinite.get_mut("q.weight").unwrap().values[0] = f32::NAN;
    assert!(matches!(
        PointerHead::from_checkpoint_tensors(nonfinite, 1.0),
        Err(HeadError::NonFiniteValue)
    ));
    assert!(matches!(
        PointerHead::from_checkpoint_tensors(small_inventory(), 0.0),
        Err(HeadError::InvalidTemperature)
    ));
    let mut overflow = small_inventory();
    overflow.get_mut("q.weight").unwrap().shape = vec![usize::MAX, 2];
    overflow.get_mut("k.weight").unwrap().shape = vec![usize::MAX, 2];
    overflow.get_mut("q.bias").unwrap().shape = vec![usize::MAX];
    overflow.get_mut("k.bias").unwrap().shape = vec![usize::MAX];
    assert!(matches!(
        PointerHead::from_checkpoint_tensors(overflow, 1.0),
        Err(HeadError::InvalidDimensions)
    ));
}

#[test]
fn scoring_rows_rejects_invalid_positions_indices_and_hidden_states() {
    let options = vec![vec![1], vec![2]];
    let encoded = encode_rows(
        &[3],
        &[Question {
            instruction: &[],
            options: &options,
        }],
        Delimiters {
            state: 100,
            question: 101,
            option_start: 102,
            option_end: 103,
            decide: 104,
        },
        16,
        128,
    )
    .unwrap();
    let row = &encoded.rows[0];
    let head = identity_head(1.0);
    let states = vec![0.0; row.token_ids.len() * 2];
    assert!(head.score_row(row, &states).is_ok());
    let mut invalid = row.clone();
    invalid.position_ids[1] += 1;
    assert_eq!(
        head.score_row(&invalid, &states),
        Err(HeadError::InvalidRow)
    );
    invalid = row.clone();
    invalid.position_ids[0] = 0;
    assert_eq!(
        head.score_row(&invalid, &states),
        Err(HeadError::InvalidRow)
    );
    invalid = row.clone();
    invalid.decision_index -= 1;
    assert_eq!(
        head.score_row(&invalid, &states),
        Err(HeadError::InvalidRow)
    );
    invalid = row.clone();
    invalid.option_end_indices[1] = invalid.option_end_indices[0];
    assert_eq!(
        head.score_row(&invalid, &states),
        Err(HeadError::InvalidRow)
    );
    invalid = row.clone();
    invalid.option_end_indices[0] = invalid.decision_index;
    assert_eq!(
        head.score_row(&invalid, &states),
        Err(HeadError::InvalidRow)
    );
    invalid = row.clone();
    invalid.option_end_indices.clear();
    assert_eq!(
        head.score_row(&invalid, &states),
        Err(HeadError::InvalidRow)
    );
    invalid.option_end_indices = vec![2; 256];
    assert_eq!(
        head.score_row(&invalid, &states),
        Err(HeadError::InvalidRow)
    );
    assert_eq!(
        head.score_row(row, &states[..states.len() - 1]),
        Err(HeadError::InvalidDimensions)
    );
    let mut nonfinite = states;
    nonfinite[0] = f32::NAN;
    assert_eq!(
        head.score_row(row, &nonfinite),
        Err(HeadError::NonFiniteValue)
    );
}

fn identity_head(temperature: f32) -> PointerHead {
    let identity = || LinearProjection::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![0.0; 2]).unwrap();
    PointerHead::new(identity(), identity(), temperature).unwrap()
}

#[test]
fn independent_questions_have_independent_softmaxes_and_temperature() {
    let head = identity_head(2.0);
    let first = head
        .score(&[2.0, 0.0], &[&[2.0, 0.0], &[0.0, 2.0]])
        .unwrap();
    let expected = 2.0_f32.sqrt();
    assert!((first.logits[0] - expected).abs() < 1e-6);
    assert_eq!(first.logits[1], 0.0);
    assert!((first.probabilities[0] - 0.804_429_7).abs() < 1e-6);
    let second = head.score(&[0.0, 1.0], &[&[0.0, 2.0]]).unwrap();
    assert_eq!(second.probabilities, [1.0]);
    assert!((first.probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
}

#[test]
fn large_logits_are_normalized_without_exponential_overflow() {
    let head = identity_head(1.0);
    let result = head
        .score(&[1000.0, 0.0], &[&[1.0, 0.0], &[0.0, 1.0]])
        .unwrap();
    assert!(result.probabilities.iter().all(|p| p.is_finite()));
    assert_eq!(result.probabilities, [1.0, 0.0]);
}

#[test]
fn malformed_weights_inputs_and_temperatures_fail_closed() {
    assert!(matches!(
        LinearProjection::new(2, 2, vec![0.0; 3], vec![0.0; 2]),
        Err(HeadError::InvalidDimensions)
    ));
    assert!(matches!(
        LinearProjection::new(2, 2, vec![0.0, 0.0, f32::NAN, 0.0], vec![0.0; 2]),
        Err(HeadError::NonFiniteValue)
    ));
    let projection = || LinearProjection::new(1, 1, vec![1.0], vec![0.0]).unwrap();
    for temperature in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(matches!(
            PointerHead::new(projection(), projection(), temperature),
            Err(HeadError::InvalidTemperature)
        ));
    }
    let mismatched = LinearProjection::new(2, 1, vec![1.0, 0.0], vec![0.0]).unwrap();
    assert!(matches!(
        PointerHead::new(projection(), mismatched, 1.0),
        Err(HeadError::InvalidDimensions)
    ));
    let head = identity_head(1.0);
    assert_eq!(head.score(&[1.0, 2.0], &[]), Err(HeadError::NoOptions));
    assert_eq!(
        head.score(&[1.0], &[&[1.0, 2.0]]),
        Err(HeadError::InvalidDimensions)
    );
    assert_eq!(
        head.score(&[1.0, 2.0], &[&[f32::INFINITY, 1.0]]),
        Err(HeadError::NonFiniteValue)
    );
    assert_eq!(
        head.score(&[f32::MAX, 0.0], &[&[f32::MAX, 0.0]]),
        Err(HeadError::NonFiniteValue)
    );
}
