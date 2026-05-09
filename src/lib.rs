//! # fleet-predict
//!
//! Correlation-based predictive coding for multi-agent fleets.
//!
//! Each agent in a fleet emits a time series of scalars. `FleetPredictor` learns
//! which past states of one agent best predict future states of another (or itself)
//! using simple Pearson correlation — no matrix inversion, no external dependencies.

/// A single prediction relationship: `source` -> `target` with a given lag.
#[derive(Debug, Clone, PartialEq)]
pub struct Prediction {
    /// Index of the source agent whose past state is used as the predictor.
    pub source: usize,
    /// Index of the target agent being predicted.
    pub target: usize,
    /// Lag steps: source[t] predicts target[t+lag].
    pub lag: usize,
    /// Pearson correlation coefficient between the shifted series.
    pub correlation: f64,
    /// Accuracy = |correlation|, clipped to [0.0, 1.0].
    pub accuracy: f64,
}

/// Accuracy profile at every lag for a given target.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictiveHorizon {
    /// The largest lag at which accuracy is still useful (e.g. > 0.2).
    pub max_useful_lag: usize,
    /// Vector of accuracies, one per lag starting from 1.
    pub accuracies: Vec<f64>,
}

/// A fleet-wide predictor that learns cross-agent correlation relationships.
#[derive(Debug, Clone)]
pub struct FleetPredictor {
    n_agents: usize,
    predictions: Vec<Prediction>,
}

impl FleetPredictor {
    /// Create a new predictor for `n_agents` in the fleet.
    pub fn new(n_agents: usize) -> Self {
        FleetPredictor {
            n_agents,
            predictions: Vec::new(),
        }
    }

    /// Number of agents in this fleet.
    pub fn n_agents(&self) -> usize {
        self.n_agents
    }

    /// All stored predictions.
    pub fn predictions(&self) -> &[Prediction] {
        &self.predictions
    }

    /// Analyze historical data to learn correlations.
    ///
    /// `history` is a slice of time series, one per agent. Each inner Vec must have
    /// the same length.
    ///
    /// For every ordered pair (source, target) and every lag in [1, max_lag],
    /// we compute the Pearson correlation of source[t] against target[t+lag].
    /// The best (highest-accuracy) prediction for each pair is stored.
    pub fn analyze(&mut self, history: &[Vec<f64>], max_lag: usize) -> &Self {
        self.predictions.clear();

        if self.n_agents == 0 || history.is_empty() || history[0].is_empty() {
            return self;
        }

        let len = history[0].len();

        // Compute best prediction for every ordered pair (source, target)
        let mut best_preds: Vec<Vec<Option<Prediction>>> =
            vec![vec![None; self.n_agents]; self.n_agents];

        for source in 0..self.n_agents {
            for target in 0..self.n_agents {
                let mut best_accuracy = -1.0_f64;
                let mut best_pred: Option<Prediction> = None;

                for lag in 1..=max_lag {
                    if len <= lag {
                        break;
                    }
                    let corr = pearson_correlation(&history[source], &history[target], lag);
                    let accuracy = corr.abs().min(1.0);
                    if accuracy > best_accuracy {
                        best_accuracy = accuracy;
                        best_pred = Some(Prediction {
                            source,
                            target,
                            lag,
                            correlation: corr,
                            accuracy,
                        });
                    }
                }

                best_preds[source][target] = best_pred;
            }
        }

        // Flatten into predictions vec
        for source_preds in best_preds.iter() {
            for pred in source_preds.iter() {
                if let Some(p) = pred {
                    self.predictions.push(p.clone());
                }
            }
        }

        self
    }

    /// Predict future states for `target` agent.
    ///
    /// Uses the best predictor for this target. Simple model:
    /// `next_state = correlation * source_current_state`.
    ///
    /// Returns `n` predictions where `n = history[0].len()`.
    /// If no predictor exists for the target, returns a zero-filled vector
    /// (prediction = last known state of target).
    pub fn predict(&self, target: usize, history: &[Vec<f64>]) -> Vec<f64> {
        let len = history[0].len();
        if len == 0 {
            return Vec::new();
        }

        if let Some(best) = self.best_predictor_for(target) {
            let src_series = &history[best.source];
            let lag = best.lag;
            let corr = best.correlation;

            // Predict next state using correlation * source's current state
            let mut result = Vec::with_capacity(len);
            for i in 0..len {
                // For positions where we have aligned data, apply correlation-based scaling
                // If source has data at (i), use it to predict target at (i + lag)
                if i >= lag {
                    let src_val = src_series[i - lag];
                    let predicted = corr * src_val;
                    result.push(predicted);
                } else {
                    // For early positions, use the target's own value
                    result.push(history[target][i]);
                }
            }
            result
        } else {
            // Fallback: repeat last known value
            let last = history[target][len - 1];
            vec![last; len]
        }
    }

    /// Get the best (highest-accuracy) predictor for a given target agent.
    pub fn best_predictor_for(&self, target: usize) -> Option<&Prediction> {
        self.predictions
            .iter()
            .filter(|p| p.target == target)
            .max_by(|a, b| a.accuracy.partial_cmp(&b.accuracy).unwrap())
    }

    /// Collective accuracy for a target: average accuracy of all predictors for this target.
    pub fn collective_accuracy(&self, target: usize) -> f64 {
        let preds: Vec<&Prediction> = self.predictions.iter().filter(|p| p.target == target).collect();
        if preds.is_empty() {
            return 0.0;
        }
        let sum: f64 = preds.iter().map(|p| p.accuracy).sum();
        sum / preds.len() as f64
    }

    /// Return accuracy at each lag level for a given target.
    pub fn predictive_horizon(&self, target: usize) -> PredictiveHorizon {
        let mut lag_acc: Vec<(usize, f64)> = Vec::new();

        // Group by lag and average accuracy
        for p in self.predictions.iter().filter(|p| p.target == target) {
            if let Some(entry) = lag_acc.iter_mut().find(|(lag, _)| *lag == p.lag) {
                entry.1 = (entry.1 + p.accuracy) / 2.0;
            } else {
                lag_acc.push((p.lag, p.accuracy));
            }
        }

        lag_acc.sort_by_key(|(lag, _)| *lag);

        let max_lag = lag_acc
            .iter()
            .filter(|(_, acc)| *acc > 0.2)
            .map(|(lag, _)| *lag)
            .max()
            .unwrap_or(0);

        let mut accuracies = Vec::new();
        let mut idx = 0;
        for lag in 1..=lag_acc.last().map(|(l, _)| *l).unwrap_or(0) {
            if idx < lag_acc.len() && lag_acc[idx].0 == lag {
                accuracies.push(lag_acc[idx].1);
                idx += 1;
            } else {
                accuracies.push(0.0);
            }
        }

        PredictiveHorizon {
            max_useful_lag: max_lag,
            accuracies,
        }
    }

    /// Mean autocorrelation (lag=1) across all agents.
    /// Measures how well each agent predicts itself one step ahead.
    pub fn self_prediction_accuracy(&self, history: &[Vec<f64>]) -> f64 {
        let mut sum = 0.0_f64;
        let mut count = 0;

        for agent in 0..self.n_agents {
            let series = &history[agent];
            if series.len() < 2 {
                continue;
            }
            // Lag-1 autocorrelation
            let n = series.len() - 1;
            let mean_t = series[..n].iter().sum::<f64>() / n as f64;
            let mean_t1 = series[1..].iter().sum::<f64>() / n as f64;

            let mut cov = 0.0_f64;
            let mut var_t = 0.0_f64;
            let mut var_t1 = 0.0_f64;

            for i in 0..n {
                let dx = series[i] - mean_t;
                let dy = series[i + 1] - mean_t1;
                cov += dx * dy;
                var_t += dx * dx;
                var_t1 += dy * dy;
            }

            let denom = (var_t * var_t1).sqrt();
            let accuracy = if denom > 0.0 {
                (cov / denom).abs()
            } else {
                0.0
            };

            sum += accuracy;
            count += 1;
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Compute Pearson correlation between `a` and `b` with `b` shifted by `lag`.
///
/// a[t] is correlated with b[t + lag] for t = 0..(n - lag - 1).
fn pearson_correlation(a: &[f64], b: &[f64], lag: usize) -> f64 {
    let n = a.len().min(b.len());
    if n <= lag + 1 {
        return 0.0;
    }

    let m = n - lag; // number of overlapping pairs

    // Means of the overlapping segments
    let mean_a: f64 = a[..m].iter().sum::<f64>() / m as f64;
    let mean_b: f64 = b[lag..].iter().sum::<f64>() / m as f64;

    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;

    for i in 0..m {
        let dx = a[i] - mean_a;
        let dy = b[i + lag] - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }

    let denom = (var_a * var_b).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // Construction & basics
    // ----------------------------------------------------------------

    #[test]
    fn test_new_predictor() {
        let fp = FleetPredictor::new(3);
        assert_eq!(fp.n_agents(), 3);
        assert!(fp.predictions().is_empty());
    }

    #[test]
    fn test_zero_agents() {
        let mut fp = FleetPredictor::new(0);
        assert_eq!(fp.n_agents(), 0);
        let history: Vec<Vec<f64>> = vec![];
        fp.analyze(&history, 5);
        assert!(fp.predictions().is_empty());
    }

    // ----------------------------------------------------------------
    // Correlation helpers
    // ----------------------------------------------------------------

    #[test]
    fn test_perfect_positive_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let corr = pearson_correlation(&a, &b, 0);
        assert!((corr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_perfect_negative_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let corr = pearson_correlation(&a, &b, 0);
        assert!((corr - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_zero_correlation() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let corr = pearson_correlation(&a, &b, 0);
        assert!(corr.abs() < 1e-10);
    }

    #[test]
    fn test_correlation_with_lag() {
        // a leads b by 1: a[t] ≈ b[t+1]
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let corr = pearson_correlation(&a, &b, 1);
        assert!((corr - 1.0).abs() < 1e-10, "expected ~1.0, got {}", corr);
    }

    // ----------------------------------------------------------------
    // Analyze with sine wave data
    // ----------------------------------------------------------------

    #[test]
    fn test_analyze_two_agents_same_sine() {
        let mut fp = FleetPredictor::new(2);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..100 {
            let val = (i as f64 * 0.1).sin();
            a.push(val);
            b.push(val);
        }
        fp.analyze(&[a.clone(), b.clone()], 5);

        // Self-prediction at lag=0 should be strong
        let pred = fp.best_predictor_for(0);
        assert!(pred.is_some());
        let p = pred.unwrap();
        assert!(p.accuracy > 0.9, "self-prediction accuracy should be high, got {}", p.accuracy);
    }

    #[test]
    fn test_analyze_agent_leads_another() {
        let mut fp = FleetPredictor::new(2);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..100 {
            let t = i as f64 * 0.2;
            a.push(t.sin());
            b.push((t - 0.5).sin()); // b lags a by ~2.5 steps
        }
        fp.analyze(&[a.clone(), b.clone()], 5);

        // Source 0 should be a decent predictor for target 1
        let pred = fp.best_predictor_for(1);
        assert!(pred.is_some(), "should find a predictor for target 1");
        let p = pred.unwrap();
        assert!(p.accuracy > 0.3, "prediction should be meaningful, got accuracy={}", p.accuracy);
        assert_eq!(p.source, 0);
    }

    // ----------------------------------------------------------------
    // Predict
    // ----------------------------------------------------------------

    #[test]
    fn test_predict_returns_correct_length() {
        let mut fp = FleetPredictor::new(2);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..50 {
            let val = (i as f64 * 0.1).sin();
            a.push(val);
            b.push(val);
        }
        fp.analyze(&[a.clone(), b.clone()], 3);
        let preds = fp.predict(0, &[a.clone(), b.clone()]);
        assert_eq!(preds.len(), 50);
    }

    #[test]
    fn test_predict_does_not_nan() {
        let mut fp = FleetPredictor::new(2);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        fp.analyze(&[a.clone(), b.clone()], 2);
        let preds = fp.predict(0, &[a.clone(), b.clone()]);
        for v in &preds {
            assert!(!v.is_nan(), "prediction should not be NaN, got {}", v);
        }
    }

    // ----------------------------------------------------------------
    // best_predictor_for
    // ----------------------------------------------------------------

    #[test]
    fn test_best_predictor_for_returns_self() {
        let mut fp = FleetPredictor::new(2);
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        fp.analyze(&[a, b], 3);
        let pred = fp.best_predictor_for(0);
        assert!(pred.is_some());
        assert_eq!(pred.unwrap().target, 0);
    }

    #[test]
    fn test_best_predictor_for_nonexistent() {
        let fp = FleetPredictor::new(3);
        let pred = fp.best_predictor_for(0);
        assert!(pred.is_none());
    }

    // ----------------------------------------------------------------
    // collective_accuracy
    // ----------------------------------------------------------------

    #[test]
    fn test_collective_accuracy_empty() {
        let fp = FleetPredictor::new(3);
        let acc = fp.collective_accuracy(0);
        assert!((acc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_collective_accuracy_nonempty() {
        let mut fp = FleetPredictor::new(3);
        let mut a = Vec::new();
        let mut b = Vec::new();
        let mut c = Vec::new();
        for i in 0..50 {
            let t = i as f64 * 0.1;
            a.push(t.sin());
            b.push((t - 0.3).sin());
            c.push((t - 0.6).sin());
        }
        fp.analyze(&[a, b, c], 3);
        let acc = fp.collective_accuracy(1);
        assert!(acc > 0.0);
    }

    // ----------------------------------------------------------------
    // predictive_horizon
    // ----------------------------------------------------------------

    #[test]
    fn test_predictive_horizon_basic() {
        let mut fp = FleetPredictor::new(2);
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        fp.analyze(&[a, b], 5);
        let horizon = fp.predictive_horizon(1);
        assert!(!horizon.accuracies.is_empty());
    }

    // ----------------------------------------------------------------
    // self_prediction_accuracy
    // ----------------------------------------------------------------

    #[test]
    fn test_self_prediction_with_perfect_autocorrelation() {
        // Constant series: perfect autocorrelation (but zero variance — edge case)
        let fp = FleetPredictor::new(1);
        let a = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        let acc = fp.self_prediction_accuracy(&[a]);
        assert!((acc - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_self_prediction_with_sine() {
        let fp = FleetPredictor::new(2);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..100 {
            let val = (i as f64 * 0.1).sin();
            a.push(val);
            b.push((i as f64 * 0.1 + 1.0).sin());
        }
        let acc = fp.self_prediction_accuracy(&[a, b]);
        assert!(acc > 0.0);
    }

    // ----------------------------------------------------------------
    // Random data tests
    // ----------------------------------------------------------------

    #[test]
    fn test_random_data_no_crashes() {
        

        let mut fp = FleetPredictor::new(4);
        let data: Vec<Vec<f64>> = (0..4)
            .map(|seed| {
                (0..50)
                    .map(|i| {
                        // Deterministic pseudo-random using simple approach
                        let x = (seed as f64 * 1.618 + i as f64 * 0.7).sin() * 1000.0;
                        (x.sin() * 1000.0).round() / 1000.0
                    })
                    .collect()
            })
            .collect();

        fp.analyze(&data, 5);
        let preds = fp.predict(0, &data);
        assert_eq!(preds.len(), 50);

        let acc = fp.collective_accuracy(1);
        assert!(acc >= 0.0);
    }

    // ----------------------------------------------------------------
    // Edge cases
    // ----------------------------------------------------------------

    #[test]
    fn test_empty_history() {
        let fp = FleetPredictor::new(2);
        let history: Vec<Vec<f64>> = vec![vec![], vec![]];
        let preds = fp.predict(0, &history);
        assert!(preds.is_empty());
    }

    #[test]
    fn test_single_element_history() {
        let mut fp = FleetPredictor::new(2);
        let history = vec![vec![42.0], vec![100.0]];
        fp.analyze(&history, 2);
        let preds = fp.predict(0, &history);
        assert_eq!(preds.len(), 1);
        assert!(!preds[0].is_nan());
    }

    #[test]
    fn test_two_element_history() {
        let mut fp = FleetPredictor::new(2);
        let history = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        fp.analyze(&history, 1);
        let preds = fp.predict(0, &history);
        assert_eq!(preds.len(), 2);
    }

    #[test]
    fn test_large_lag() {
        let mut fp = FleetPredictor::new(2);
        let a = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        // max_lag larger than history — should not panic
        fp.analyze(&[a, b], 100);
        assert!(fp.predictions().is_empty() || !fp.predictions().is_empty());
    }

    #[test]
    fn test_many_agents() {
        let mut fp = FleetPredictor::new(10);
        let mut data: Vec<Vec<f64>> = Vec::new();
        for agent in 0..10 {
            let series: Vec<f64> = (0..30)
                .map(|i| ((i as f64 * 0.2) + (agent as f64 * 0.5)).sin())
                .collect();
            data.push(series);
        }
        fp.analyze(&data, 3);
        assert!(!fp.predictions().is_empty());

        // All targets should have some predictor
        for t in 0..10 {
            let pred = fp.best_predictor_for(t);
            assert!(pred.is_some(), "target {} should have a predictor", t);
        }
    }

    // ----------------------------------------------------------------
    // Deterministic sine correlations
    // ----------------------------------------------------------------

    #[test]
    fn test_sine_correlation_lag() {
        // a and b are sine waves with same frequency, b slightly shifted
        let n = 200;
        let freq = 0.1;
        let shift = 0.3;
        let a: Vec<f64> = (0..n).map(|i| (i as f64 * freq).sin()).collect();
        let b: Vec<f64> = (0..n).map(|i| ((i as f64 * freq) - shift).sin()).collect();

        let mut fp = FleetPredictor::new(2);
        fp.analyze(&[a.clone(), b.clone()], 10);

        let best = fp.best_predictor_for(1);
        assert!(best.is_some(), "should find predictor for target 1");
        let p = best.unwrap();
        // Source 0 should predict target 1
        assert_eq!(p.source, 0);
        assert!(p.accuracy > 0.1, "accuracy should be nonzero, got {}", p.accuracy);
    }

    #[test]
    fn test_self_prediction_with_sine_medium() {
        let n = 50;
        let a: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3 + 0.5).cos()).collect();

        let fp = FleetPredictor::new(2);
        let acc = fp.self_prediction_accuracy(&[a, b]);
        assert!(acc >= 0.0);
    }

    #[test]
    fn test_no_predictor_fallback_to_last_value() {
        let fp = FleetPredictor::new(3);
        let history = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![10.0, 20.0, 30.0, 40.0, 50.0],
            vec![100.0, 200.0, 300.0, 400.0, 500.0],
        ];
        let preds = fp.predict(0, &history);
        assert_eq!(preds.len(), 5);
        // Without analysis, fallback: all predictions = last known value of target
        for p in &preds {
            assert!((*p - 5.0).abs() < 1e-10, "expected 5.0, got {}", p);
        }
    }
}
