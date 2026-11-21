//! Kev pointer-head reference implementation.
//!
//! Given the final hidden state of a decision token and one hidden state per
//! option's closing token, this computes the two learned linear projections,
//! scaled dot products, temperature adjustment and per-question softmax.
//! It does not load checkpoints, run Qwen, tokenize text or implement a
//! Magnetar Component Model export.

use std::error::Error;
use std::fmt;

pub mod encoding;

/// Row-major affine transform, with weights shaped `[output_dim, input_dim]`.
#[derive(Debug, Clone)]
pub struct LinearProjection {
    input_dim: usize,
    output_dim: usize,
    weights: Vec<f32>,
    bias: Vec<f32>,
}

impl LinearProjection {
    /// Construct a projection, rejecting inconsistent or non-finite weights.
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, HeadError> {
        if input_dim == 0 || output_dim == 0 {
            return Err(HeadError::InvalidDimensions);
        }
        let size = input_dim
            .checked_mul(output_dim)
            .ok_or(HeadError::InvalidDimensions)?;
        if weights.len() != size || bias.len() != output_dim {
            return Err(HeadError::InvalidDimensions);
        }
        if weights.iter().chain(&bias).any(|value| !value.is_finite()) {
            return Err(HeadError::NonFiniteValue);
        }
        Ok(Self {
            input_dim,
            output_dim,
            weights,
            bias,
        })
    }

    fn apply(&self, input: &[f32]) -> Result<Vec<f32>, HeadError> {
        if input.len() != self.input_dim {
            return Err(HeadError::InvalidDimensions);
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(HeadError::NonFiniteValue);
        }
        self.weights
            .chunks_exact(self.input_dim)
            .zip(&self.bias)
            .map(|(row, bias)| {
                let result = row
                    .iter()
                    .zip(input)
                    .fold(*bias, |sum, (weight, value)| sum + weight * value);
                if result.is_finite() {
                    Ok(result)
                } else {
                    Err(HeadError::NonFiniteValue)
                }
            })
            .collect()
    }
}

/// Kev's query and key projections plus inference-time temperature.
#[derive(Debug, Clone)]
pub struct PointerHead {
    query: LinearProjection,
    key: LinearProjection,
    temperature: f32,
}

/// Calibrated logits and probabilities, in the caller's option order.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionScores {
    pub logits: Vec<f32>,
    pub probabilities: Vec<f32>,
}

impl PointerHead {
    /// Both projections must map the same hidden dimension to the same
    /// nonzero pointer dimension; temperature must be finite and positive.
    pub fn new(
        query: LinearProjection,
        key: LinearProjection,
        temperature: f32,
    ) -> Result<Self, HeadError> {
        if query.input_dim != key.input_dim || query.output_dim != key.output_dim {
            return Err(HeadError::InvalidDimensions);
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err(HeadError::InvalidTemperature);
        }
        Ok(Self {
            query,
            key,
            temperature,
        })
    }

    /// Score one question. `decision` is its `<decide>` hidden state;
    /// `options` contains the `</opt>` hidden state of each option, in order.
    ///
    /// The caller is responsible for producing correctly isolated question
    /// states with a compatible backbone. No confidence calibration is
    /// claimed beyond applying the caller-supplied checkpoint temperature.
    pub fn score(&self, decision: &[f32], options: &[&[f32]]) -> Result<DecisionScores, HeadError> {
        if options.is_empty() {
            return Err(HeadError::NoOptions);
        }
        let query = self.query.apply(decision)?;
        let scale = (self.query.output_dim as f32).sqrt().recip();
        let mut logits = Vec::with_capacity(options.len());
        for option in options {
            let key = self.key.apply(option)?;
            let dot = key.iter().zip(&query).map(|(k, q)| k * q).sum::<f32>();
            let logit = dot * scale / self.temperature;
            if !logit.is_finite() {
                return Err(HeadError::NonFiniteValue);
            }
            logits.push(logit);
        }
        // Max subtraction keeps the exponentials bounded for large logits.
        let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut probabilities: Vec<f32> = logits.iter().map(|z| (z - maximum).exp()).collect();
        let normalizer: f32 = probabilities.iter().sum();
        for probability in &mut probabilities {
            *probability /= normalizer;
        }
        Ok(DecisionScores {
            logits,
            probabilities,
        })
    }
}

/// Invalid weights, hidden states, or scoring parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeadError {
    InvalidDimensions,
    InvalidTemperature,
    NonFiniteValue,
    NoOptions,
}

impl fmt::Display for HeadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            Self::InvalidDimensions => "inconsistent or empty tensor dimensions",
            Self::InvalidTemperature => "temperature must be finite and positive",
            Self::NonFiniteValue => "non-finite input or intermediate value",
            Self::NoOptions => "at least one option is required",
        };
        formatter.write_str(message)
    }
}

impl Error for HeadError {}
