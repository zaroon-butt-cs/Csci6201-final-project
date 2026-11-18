import { useEffect, useMemo, useRef, useState } from 'react'
import ChoiceField from './components/ChoiceField'
import Drivers from './components/Drivers'
import Gauge from './components/Gauge'
import NumberField from './components/NumberField'
import ProjectInfo from './components/ProjectInfo'
import { FIELD_INFO, GROUPS } from './lib/features'
import {
  bandOf,
  defaultValues,
  featureByKey,
  model,
  percentileOf,
  predict,
} from './lib/predict'
import { useAnimatedNumber } from './lib/useAnimatedNumber'

const TYPICAL_ERROR = Math.round(model.performance.cv_mae)
const EXPLAINED = Math.round(model.performance.cv_r2 * 100)

export default function App() {
  const [values, setValues] = useState(defaultValues)
  const [example, setExample] = useState(null) // index of the loaded example patient
  const resultRef = useRef(null)
  const [resultVisible, setResultVisible] = useState(true)

  const { score, contributions } = useMemo(() => predict(values), [values])
  const shown = useAnimatedNumber(score)
  const band = bandOf(score)
  const percentile = percentileOf(score)
  const effects = Object.fromEntries(contributions.map((c) => [c.key, c.effect]))

  // On small screens a bar with the live score appears while the result
  // card is scrolled out of view.
  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => setResultVisible(entry.isIntersecting),
      { threshold: 0.15 },
    )
    observer.observe(resultRef.current)
    return () => observer.disconnect()
  }, [])

  const setField = (key) => (update) => {
    setExample(null)
    setValues((current) => ({ ...current, [key]: update(current[key]) }))
  }

  const loadExample = (index) => {
    setExample(index)
    setValues(model.examples[index].values)
  }

  const reset = () => {
    setExample(null)
    setValues(defaultValues())
  }

  const actual = example === null ? null : model.examples[example].actual

  return (
    <div className="app">
      <header className="masthead">
        <div className="masthead-row">
          <p className="eyebrow">CSCI 6201 · Final Project · Zaroon Butt</p>
          <ProjectInfo />
        </div>
        <h1>Diabetes progression predictor</h1>
        <p className="lead">
          Enter a patient&rsquo;s baseline measurements to estimate how far their diabetes
          is likely to progress over the next year. The estimate updates as you change
          each value.
        </p>
      </header>

      <main className="layout">
        <section className="inputs" aria-label="Patient measurements">
          <div className="toolbar">
            <span className="toolbar-label">Try a real patient</span>
            <div className="chips">
              {model.examples.map((item, index) => (
                <button
                  key={item.label}
                  type="button"
                  className={`chip ${example === index ? 'is-on' : ''}`}
                  aria-pressed={example === index}
                  onClick={() => loadExample(index)}
                >
                  {item.label}
                </button>
              ))}
            </div>
            <button type="button" className="reset" onClick={reset}>
              Reset
            </button>
          </div>

          {GROUPS.map((group) => (
            <fieldset key={group.id} className="group">
              <legend>{group.title}</legend>
              <p className="group-hint">{group.hint}</p>
              <div className="fields">
                {group.fields.map((key) => {
                  const info = FIELD_INFO[key]
                  const common = {
                    id: `field-${key}`,
                    info,
                    value: values[key],
                    effect: effects[key],
                    onChange: setField(key),
                  }
                  return info.options ? (
                    <ChoiceField key={key} {...common} />
                  ) : (
                    <NumberField key={key} {...common} stats={featureByKey[key]} />
                  )
                })}
              </div>
            </fieldset>
          ))}
        </section>

        <aside className="result" ref={resultRef} aria-label="Prediction">
          <div className="result-card">
            <p className="result-title">Predicted progression after one year</p>

            <div className="gauge-wrap">
              <Gauge score={shown} error={TYPICAL_ERROR} band={band.id} />
              <div className="score">
                <span className="score-number">{Math.round(shown)}</span>
                <span className="score-caption">
                  on a scale of {model.target.min}&ndash;{model.target.max}
                </span>
              </div>
            </div>

            <p className="verdict">
              <span className={`band band-${band.id}`}>{band.label}</span>
              Higher than about <strong>{percentile}%</strong> of the {model.trained_on}{' '}
              patients in the study data.
            </p>

            <dl className="facts">
              <div>
                <dt>Typical error</dt>
                <dd>±{TYPICAL_ERROR} points</dd>
              </div>
              <div>
                <dt>Typical range</dt>
                <dd>
                  {Math.round(score - TYPICAL_ERROR)}&ndash;{Math.round(score + TYPICAL_ERROR)}
                </dd>
              </div>
              {actual !== null && (
                <div className="fact-actual">
                  <dt>Recorded for this patient</dt>
                  <dd>
                    {actual}{' '}
                    <span className="fact-note">
                      (model off by {Math.abs(Math.round(score - actual))})
                    </span>
                  </dd>
                </div>
              )}
            </dl>

            <section className="drivers-block">
              <h2>What moves this prediction</h2>
              <p className="drivers-hint">
                Points added or removed by each value, compared with the average patient
                (score {Math.round(model.intercept)}).
              </p>
              <div className="drivers-legend" aria-hidden="true">
                <span>&larr; lowers</span>
                <span>raises &rarr;</span>
              </div>
              <Drivers contributions={contributions} />
            </section>
          </div>

          <p className="model-note">
            <strong>Model:</strong> {model.model}, selected from five regression models by
            repeated cross-validation. It explains about {EXPLAINED}% of the differences
            between patients, so treat the result as a rough estimate. Low, Normal and High are
            the lower, middle and upper third of the study patients, not medical categories.
          </p>
        </aside>
      </main>

      <footer className="footer">
        Student project built on the scikit-learn diabetes dataset ({model.trained_on}{' '}
        patients). For learning only. It is not medical advice and must not be used to
        make health decisions.
      </footer>

      <div className={`mobile-bar ${resultVisible ? '' : 'is-shown'}`} aria-hidden={resultVisible}>
        <div>
          <span className="mobile-score">{Math.round(shown)}</span>
          <span className={`band band-${band.id}`}>{band.label}</span>
        </div>
        <button
          type="button"
          tabIndex={resultVisible ? -1 : 0}
          onClick={() => resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })}
        >
          See details
        </button>
      </div>
    </div>
  )
}
