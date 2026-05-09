# fleet-predict ⚒️

**Correlation-based predictive coding for multi-agent fleets.**

A lightweight Rust library that learns how agents in a fleet predict each other's future states using simple Pearson correlation — no matrix inversion, no external dependencies.

## Motivation

In multi-agent systems, agents don't operate in isolation. If you can predict what agent B will do based on what agent A is doing now, you can:

- Detect anomalous agent behavior
- Pre-allocate resources based on predicted demand
- Identify information flow patterns in the fleet
- Build consensus models from cross-prediction accuracy

`fleet-predict` provides the foundation for this: given a time series per agent, it finds the best lagged correlation between every ordered pair.

## Types

| Type | Description |
|------|-------------|
| `Prediction` | A single relationship: `source` predicts `target` at a given `lag` with `correlation` and `accuracy` |
| `PredictiveHorizon` | Accuracy profile across all lags for one target |
| `FleetPredictor` | Main API — learns and queries fleet-wide predictions |

## API

```rust
// Create
let mut predictor = FleetPredictor::new(n_agents);

// Learn correlations from historical data
predictor.analyze(&history, max_lag);

// Predict future states for a target
let future = predictor.predict(target, &history);

// Find the best predictor for a target
let best = predictor.best_predictor_for(target);

// Average accuracy of all predictors for a target
let acc = predictor.collective_accuracy(target);

// Accuracy at each lag level for a target
let horizon = predictor.predictive_horizon(target);

// How well agents predict themselves (lag=1 autocorrelation)
let self_acc = predictor.self_prediction_accuracy(&history);
```

## How It Works

For each ordered pair (source, target) and each lag 1..max_lag:

1. Align the series: `source[t]` correlates with `target[t + lag]`
2. Compute Pearson correlation coefficient over the overlapping segment
3. Store the lag with the highest `|correlation|` for each pair

Prediction is simple: `next_state = correlation × source_current_state`.

## Example

```rust
use fleet_predict::FleetPredictor;

// Two agents with sine-wave behavior, one slightly lagging
let a: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
let b: Vec<f64> = (0..100).map(|i| ((i as f64 * 0.1) - 0.5).sin()).collect();

let mut predictor = FleetPredictor::new(2);
predictor.analyze(&[a, b], 5);

if let Some(best) = predictor.best_predictor_for(1) {
    println!("Agent 0 predicts Agent 1 at lag {} with accuracy {:.2}",
        best.lag, best.accuracy);
}
```

## Testing

```
cargo test    # 26 tests
```

## License

MIT
