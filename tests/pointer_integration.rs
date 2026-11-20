use magnetar_component_kev::{HeadError, LinearProjection, PointerHead};
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
    let query = LinearProjection::new(
        reference.input_dim,
        reference.pointer_dim,
        reference.query_weights,
        reference.query_bias,
    )
    .unwrap();
    let key = LinearProjection::new(
        reference.input_dim,
        reference.pointer_dim,
        reference.key_weights,
        reference.key_bias,
    )
    .unwrap();
    let head = PointerHead::new(query, key, reference.temperature).unwrap();
    for (index, question) in reference.questions.iter().enumerate() {
        let options: Vec<&[f32]> = question.options.iter().map(Vec::as_slice).collect();
        let actual = head.score(&question.decision, &options).unwrap();
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
