// The prediction formula, identical to the scikit-learn pipeline
// (StandardScaler -> linear model) that was exported to model.json:
//
//   z_i        = (value_i - mean_i) / scale_i
//   prediction = intercept + sum(coefficient_i * z_i)

import model from '../data/model.json'

export { model }

export const featureByKey = Object.fromEntries(model.features.map((f) => [f.key, f]))

export function predict(values) {
  let score = model.intercept
  const contributions = model.features.map((f) => {
    const z = (values[f.key] - f.mean) / f.scale
    const effect = f.coefficient * z // how far this input moves the score from average
    score += effect
    return { key: f.key, effect }
  })
  return { score, contributions }
}

// Share of the 442 dataset patients whose recorded progression is at or
// below this score (0-100).
export function percentileOf(score) {
  const p = model.target.percentiles
  let i = 0
  while (i < 100 && p[i + 1] <= score) i += 1
  return score < p[0] ? 0 : i
}

// Thirds of the dataset: lower, middle, upper.
export const BAND_LIMITS = [model.target.percentiles[33], model.target.percentiles[67]]

export function bandOf(score) {
  if (score < BAND_LIMITS[0]) return { id: 'low', label: 'Low' }
  if (score < BAND_LIMITS[1]) return { id: 'mid', label: 'Normal' }
  return { id: 'high', label: 'High' }
}

export function defaultValues() {
  // Start from the typical (median) patient
  return Object.fromEntries(model.features.map((f) => [f.key, f.median]))
}
