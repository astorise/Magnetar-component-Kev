use std::collections::BTreeMap;

use magnetar_component_kev::qwen35_adapter::{
    Kev08bProfile, LayerKind, ProfileError, TensorInfo, apply_lora_f32, merge_lora_f32,
};
use serde::Deserialize;

const BASE_CONFIG: &[u8] = include_bytes!("fixtures/qwen3.5-0.8b-base-config.json");
const ADAPTER_CONFIG: &[u8] = include_bytes!("fixtures/kev-0.8b-adapter-config.json");

fn profile() -> Kev08bProfile {
    Kev08bProfile::from_configs(BASE_CONFIG, ADAPTER_CONFIG).unwrap()
}

fn inventory(bytes: &[u8]) -> BTreeMap<String, TensorInfo> {
    #[derive(Deserialize)]
    struct HeaderEntry {
        dtype: String,
        shape: Vec<usize>,
    }
    serde_json::from_slice::<BTreeMap<String, HeaderEntry>>(bytes)
        .unwrap()
        .into_iter()
        .map(|(name, item)| {
            (
                name,
                TensorInfo {
                    dtype: item.dtype,
                    shape: item.shape,
                },
            )
        })
        .collect()
}

#[test]
fn pinned_hybrid_config_and_all_real_checkpoint_tensor_shapes_match() {
    let profile = profile();
    assert_eq!(profile.layers().len(), 24);
    assert_eq!(profile.targets().len(), 186);
    assert_eq!(profile.scale(), 2.0);
    for (index, kind) in profile.layers().iter().enumerate() {
        assert_eq!(
            *kind,
            if index % 4 == 3 {
                LayerKind::FullAttention
            } else {
                LayerKind::LinearAttention
            }
        );
    }
    let adapter = inventory(include_bytes!("fixtures/kev-0.8b-adapter-header.json"));
    let base = inventory(include_bytes!("fixtures/qwen3.5-0.8b-base-header.json"));
    assert_eq!(adapter.len(), 372);
    profile.validate_inventory(&adapter).unwrap();
    profile.validate_base_inventory(&base).unwrap();

    let first = &profile.targets()[0];
    assert_eq!(
        first.a_checkpoint_name,
        "base_model.model.layers.0.linear_attn.in_proj_a.lora_A.weight"
    );
    assert_eq!(
        first.base_checkpoint_name,
        "model.language_model.layers.0.linear_attn.in_proj_a.weight"
    );
    let gated_query = profile
        .targets()
        .iter()
        .find(|target| {
            target
                .base_checkpoint_name
                .ends_with("layers.3.self_attn.q_proj.weight")
        })
        .unwrap();
    assert_eq!(
        (gated_query.input_dim, gated_query.output_dim),
        (1024, 4096)
    );
}

#[test]
fn unsupported_configs_and_incorrect_inventories_are_rejected() {
    let mut adapter_config: serde_json::Value = serde_json::from_slice(ADAPTER_CONFIG).unwrap();
    adapter_config["r"] = 8.into();
    assert!(matches!(
        Kev08bProfile::from_configs(BASE_CONFIG, &serde_json::to_vec(&adapter_config).unwrap()),
        Err(ProfileError::UnsupportedConfig)
    ));
    let mut base_config: serde_json::Value = serde_json::from_slice(BASE_CONFIG).unwrap();
    base_config["text_config"]["layer_types"][0] = "full_attention".into();
    assert!(matches!(
        Kev08bProfile::from_configs(&serde_json::to_vec(&base_config).unwrap(), ADAPTER_CONFIG),
        Err(ProfileError::UnsupportedConfig)
    ));

    let profile = profile();
    let adapter = inventory(include_bytes!("fixtures/kev-0.8b-adapter-header.json"));
    let name = &profile.targets()[0].a_checkpoint_name;
    let mut wrong = adapter.clone();
    wrong.get_mut(name).unwrap().dtype = "BF16".into();
    assert_eq!(
        profile.validate_inventory(&wrong),
        Err(ProfileError::InventoryMismatch)
    );
    let mut wrong = adapter.clone();
    wrong.get_mut(name).unwrap().shape.reverse();
    assert_eq!(
        profile.validate_inventory(&wrong),
        Err(ProfileError::InventoryMismatch)
    );
    let mut wrong = adapter.clone();
    wrong.remove(name);
    assert_eq!(
        profile.validate_inventory(&wrong),
        Err(ProfileError::InventoryMismatch)
    );
    let mut wrong = adapter;
    wrong.insert(
        "extra".into(),
        TensorInfo {
            dtype: "F32".into(),
            shape: vec![1],
        },
    );
    assert_eq!(
        profile.validate_inventory(&wrong),
        Err(ProfileError::InventoryMismatch)
    );
    let mut base = inventory(include_bytes!("fixtures/qwen3.5-0.8b-base-header.json"));
    base.get_mut(&profile.targets()[0].base_checkpoint_name)
        .unwrap()
        .shape
        .reverse();
    assert_eq!(
        profile.validate_base_inventory(&base),
        Err(ProfileError::InventoryMismatch)
    );
}

#[derive(Deserialize)]
struct Reference {
    module: String,
    rank: usize,
    input: Vec<f32>,
    base: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    output: Vec<f32>,
}

#[test]
fn real_adapter_factor_slices_match_independent_pytorch() {
    let cases: Vec<Reference> =
        serde_json::from_slice(include_bytes!("fixtures/kev-0.8b-lora-reference.json")).unwrap();
    assert_eq!(cases.len(), 4);
    for case in cases {
        let actual = apply_lora_f32(
            &case.input,
            &case.base,
            &case.a,
            &case.b,
            case.rank,
            profile().scale(),
        )
        .unwrap();
        assert_eq!(actual.len(), case.output.len());
        for (value, expected) in actual.iter().zip(&case.output) {
            assert!(
                (value - expected).abs() < 2e-6,
                "{}: {value} != {expected}",
                case.module
            );
        }
    }
}

#[test]
fn invalid_factors_and_non_finite_results_fail_without_mutating_base() {
    let base = [1.0, 2.0];
    let input = [3.0, 4.0];
    let a = [1.0, 2.0];
    let b = [2.0, 3.0];
    assert_eq!(
        apply_lora_f32(&input, &base, &a, &b, 1, 2.0).unwrap(),
        [45.0, 68.0]
    );
    assert_eq!(
        apply_lora_f32(&input, &base, &a[..1], &b, 1, 2.0),
        Err(ProfileError::InvalidDimensions)
    );
    assert_eq!(
        apply_lora_f32(&input, &base, &a, &b, 0, 2.0),
        Err(ProfileError::InvalidDimensions)
    );
    assert_eq!(
        apply_lora_f32(&input, &base, &a, &b, 1, f32::NAN),
        Err(ProfileError::InvalidDimensions)
    );
    assert_eq!(
        apply_lora_f32(&[f32::NAN, 4.0], &base, &a, &b, 1, 2.0),
        Err(ProfileError::NonFiniteValue)
    );
    assert_eq!(
        apply_lora_f32(&[f32::MAX, 4.0], &base, &a, &b, 1, 2.0),
        Err(ProfileError::NonFiniteValue)
    );
    assert_eq!(base, [1.0, 2.0]);
}

#[test]
fn merges_checkpoint_order_factors_without_mutating_inputs() {
    let base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a = [1.0, 0.0, -1.0, 0.0, 2.0, 1.0];
    let b = [2.0, -1.0, 0.5, 3.0];
    assert_eq!(
        merge_lora_f32(&base, &a, &b, 3, 2, 2, 0.5).unwrap(),
        [2.0, 1.0, 1.5, 4.25, 8.0, 7.25]
    );
    assert_eq!(base, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        merge_lora_f32(&base, &a[..5], &b, 3, 2, 2, 0.5),
        Err(ProfileError::InvalidDimensions)
    );
    assert_eq!(
        merge_lora_f32(&base, &a, &b, 3, 2, 0, 0.5),
        Err(ProfileError::InvalidDimensions)
    );
    assert_eq!(
        merge_lora_f32(&base, &a, &b, usize::MAX, 2, 2, 0.5),
        Err(ProfileError::InvalidDimensions)
    );
    assert_eq!(
        merge_lora_f32(&base, &a, &b, 3, 2, 2, f32::NAN),
        Err(ProfileError::InvalidDimensions)
    );
    let mut nonfinite = base;
    nonfinite[0] = f32::INFINITY;
    assert_eq!(
        merge_lora_f32(&nonfinite, &a, &b, 3, 2, 2, 0.5),
        Err(ProfileError::NonFiniteValue)
    );
    assert_eq!(
        merge_lora_f32(&base, &[f32::MAX; 6], &b, 3, 2, 2, 2.0),
        Err(ProfileError::NonFiniteValue)
    );
}

/// Uses the complete real A/B factors rather than just the checked-in slices.
#[test]
#[ignore = "requires the pinned Kev checkpoint and a generated full PyTorch reference"]
fn full_real_adapter_projections_match_pytorch() {
    use sha2::{Digest, Sha256};

    let dir = std::path::PathBuf::from(
        std::env::var_os("MAGNETAR_KEV_LORA_DIR").expect("set MAGNETAR_KEV_LORA_DIR"),
    );
    let bytes = std::fs::read(dir.join("kev-adapter.safetensors")).unwrap();
    let digest: String = Sha256::digest(&bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(
        digest,
        "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a"
    );
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let payload = &bytes[8 + header_len..];
    let read_factor = |name: &str, shape: &[usize]| {
        let entry = &header[name];
        assert_eq!(entry["dtype"], "F32");
        assert_eq!(entry["shape"], serde_json::json!(shape));
        let start = entry["data_offsets"][0].as_u64().unwrap() as usize;
        let end = entry["data_offsets"][1].as_u64().unwrap() as usize;
        let raw = &payload[start..end];
        assert_eq!(raw.len(), shape.iter().product::<usize>() * 4);
        raw.as_chunks::<4>()
            .0
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<_>>()
    };
    #[derive(Deserialize)]
    struct FullReference {
        module: String,
        input: Vec<f32>,
        base: Vec<f32>,
        output: Vec<f32>,
    }
    let cases: Vec<FullReference> =
        serde_json::from_slice(&std::fs::read(dir.join("kev-lora-full-reference.json")).unwrap())
            .unwrap();
    assert_eq!(cases.len(), 2);
    let profile = profile();
    let mut max_error = 0.0_f32;
    for case in cases {
        let target = profile
            .targets()
            .iter()
            .find(|target| {
                target.base_checkpoint_name
                    == format!("model.language_model.{}.weight", case.module)
            })
            .unwrap();
        assert_eq!(case.input.len(), target.input_dim);
        assert_eq!(case.base.len(), target.output_dim);
        let a = read_factor(&target.a_checkpoint_name, &[target.rank, target.input_dim]);
        let b = read_factor(&target.b_checkpoint_name, &[target.output_dim, target.rank]);
        let actual = apply_lora_f32(
            &case.input,
            &case.base,
            &a,
            &b,
            target.rank,
            profile.scale(),
        )
        .unwrap();
        let merged = merge_lora_f32(
            &vec![0.0; target.output_dim * target.input_dim],
            &a,
            &b,
            target.input_dim,
            target.output_dim,
            target.rank,
            profile.scale(),
        )
        .unwrap();
        let merged_output: Vec<f32> = merged
            .chunks_exact(target.input_dim)
            .zip(&case.base)
            .map(|(row, base)| {
                base + row
                    .iter()
                    .zip(&case.input)
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>()
            })
            .collect();
        assert_eq!(actual.len(), case.output.len());
        for values in [&actual, &merged_output] {
            for (&value, &expected) in values.iter().zip(&case.output) {
                max_error = max_error.max((value - expected).abs());
                assert!(
                    (value - expected).abs() < 2e-6 + 1e-6 * expected.abs(),
                    "{}: {value} != {expected}",
                    case.module
                );
            }
        }
    }
    eprintln!("maximum absolute error: {max_error}");
}
