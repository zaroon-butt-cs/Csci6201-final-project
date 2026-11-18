import { useEffect, useRef } from 'react'
import { FIELD_INFO } from '../lib/features'
import { model } from '../lib/predict'

const { protocol, comparison, target, performance } = model
const selected = comparison.find((row) => row.selected)
const rounds = protocol.cv_folds * protocol.cv_repeats
const tiedNames = comparison.filter((row) => row.tied).map((row) => row.model)
const worseNames = comparison.filter((row) => !row.tied).map((row) => row.model)

// Features with the largest standardised coefficients (strongest influence)
const strongest = [...model.features]
  .sort((a, b) => Math.abs(b.coefficient) - Math.abs(a.coefficient))
  .slice(0, 4)

const listOf = (names) =>
  names.length <= 1 ? names.join('') : `${names.slice(0, -1).join(', ')} and ${names.at(-1)}`

const pValue = (p) => (p === null ? 'top' : p < 0.001 ? '< 0.001' : p.toFixed(3))
const verdict = (row) => (row.selected ? 'Selected' : row.tied ? 'Tied' : 'Worse')

/** "Project info" button plus the dialog with the project's key facts. */
export default function ProjectInfo() {
  const dialog = useRef(null)

  // Opening the page with #info in the address shows the summary straight away
  useEffect(() => {
    if (window.location.hash === '#info' && !dialog.current.open) dialog.current.showModal()
  }, [])

  return (
    <>
      <button type="button" className="info-button" onClick={() => dialog.current.showModal()}>
        <svg viewBox="0 0 20 20" aria-hidden="true">
          <circle cx="10" cy="10" r="8" />
          <path d="M10 9v5M10 6.2v.1" />
        </svg>
        Project info
      </button>

      <dialog
        ref={dialog}
        className="info-dialog"
        aria-labelledby="info-title"
        // a click on the dark backdrop (the dialog element itself) closes it
        onClick={(event) => event.target === dialog.current && dialog.current.close()}
      >
        <div className="info-body">
          <header className="info-head">
            <div>
              <p className="eyebrow">CSCI 6201 · Final Project · Zaroon Butt</p>
              <h2 id="info-title">Project summary</h2>
            </div>
            <button type="button" className="info-close" aria-label="Close"
              onClick={() => dialog.current.close()}>
              <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 3l10 10M13 3L3 13" /></svg>
            </button>
          </header>

          <div className="info-stats">
            <div><span>Selected model</span><strong>{model.model}</strong></div>
            <div><span>R² (cross-validated)</span><strong>{performance.cv_r2.toFixed(3)}</strong></div>
            <div><span>MAE</span><strong>{performance.cv_mae.toFixed(1)}</strong></div>
            <div><span>RMSE</span><strong>{performance.cv_rmse.toFixed(1)}</strong></div>
          </div>

          <section>
            <h3>Goal</h3>
            <ul>
              <li>Predict how far a patient&rsquo;s diabetes progresses one year after a baseline visit.</li>
              <li>Problem type: <b>regression</b> (the target is a number, not a class).</li>
              <li>Question: which regression model predicts it best, and how well?</li>
            </ul>
          </section>

          <section>
            <h3>Dataset</h3>
            <ul>
              <li>scikit-learn Diabetes dataset (Efron, Hastie, Johnstone and Tibshirani, 2004).</li>
              <li><b>{model.trained_on} patients</b>, {model.features.length} features, no missing values.</li>
              <li>Features: {model.features.map((f) => FIELD_INFO[f.key].label).join(', ')}.</li>
              <li>
                Target: a disease-progression score from {target.min} to {target.max} (mean{' '}
                {target.mean.toFixed(0)}). Higher means more progression.
              </li>
              <li>Loaded with <code>scaled=False</code>, so values are in their original units.</li>
            </ul>
          </section>

          <section>
            <h3>Training and evaluation</h3>
            <ul>
              <li>
                <b>Train / test split:</b> {Math.round((1 - protocol.test_size) * 100)}% training (
                {protocol.n_train} patients) and {Math.round(protocol.test_size * 100)}% testing (
                {protocol.n_test} patients), <code>random_state={protocol.random_state}</code>.
              </li>
              <li>
                <b>Cross-validation:</b> {protocol.cv_folds} folds repeated {protocol.cv_repeats} times
                = <b>{rounds} rounds</b> on all {model.trained_on} patients. Every model is scored on
                the same rounds.
              </li>
              <li>
                <b>Tuning:</b> grid search with {protocol.cv_folds}-fold cross-validation inside each
                round (nested), so the test part of a round is never used to choose settings.
              </li>
              <li>
                <b>Scaling:</b> StandardScaler inside a pipeline for the linear models, fitted on
                training data only (no data leakage). Tree models need no scaling.
              </li>
              <li>
                <b>Significance test:</b> corrected resampled t-test (Nadeau and Bengio, 2003) at
                the {protocol.significance_level} level.
              </li>
              <li>
                <b>Selection rule:</b> highest mean R²; models that are not significantly worse
                count as tied; the simplest tied model is chosen.
              </li>
            </ul>
          </section>

          <section>
            <h3>Metrics</h3>
            <ul>
              <li><b>R²</b>: share of the differences between patients that the model explains. 1.0 is perfect, 0 is no better than guessing the average.</li>
              <li><b>MAE</b>: average size of the error, in score points. Lower is better.</li>
              <li><b>RMSE</b>: like MAE but large errors count more. Lower is better.</li>
            </ul>
          </section>

          <section>
            <h3>Results for all five models</h3>
            <div className="info-table">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>R²</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>Test R²</th>
                    <th>p vs top</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {comparison.map((row) => (
                    <tr key={row.model} className={row.selected ? 'is-selected' : ''}>
                      <th scope="row">{row.model}</th>
                      <td>{row.cv_r2.toFixed(3)}</td>
                      <td>{row.cv_mae.toFixed(1)}</td>
                      <td>{row.cv_rmse.toFixed(1)}</td>
                      <td>{row.test_r2.toFixed(3)}</td>
                      <td>{pValue(row.p_vs_top)}</td>
                      <td>{verdict(row)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="info-caption">
              R², MAE and RMSE are averages over the {rounds} cross-validation rounds. Test R² is
              the single held-out split of {protocol.n_test} patients.
            </p>
            <ul>
              <li>{listOf(tiedNames)} are statistically tied.</li>
              {worseNames.length > 0 && (
                <li>
                  {listOf(worseNames)} {worseNames.length === 1 ? 'is' : 'are'} significantly worse,
                  even after tuning.
                </li>
              )}
              <li>
                <b>{model.model} was selected</b>: tied for the best accuracy and the simplest of
                the tied models.
              </li>
            </ul>
          </section>

          <section>
            <h3>The selected model</h3>
            <ul>
              <li>
                {model.model}: cross-validated R² {selected.cv_r2.toFixed(3)} (± {selected.cv_r2_std.toFixed(3)}),
                MAE {selected.cv_mae.toFixed(1)}, RMSE {selected.cv_rmse.toFixed(1)}; held-out test
                R² {selected.test_r2.toFixed(3)}, MAE {selected.test_mae.toFixed(1)}.
              </li>
              <li>
                Formula: <code>prediction = intercept + Σ coefficient × (value − mean) ÷ std</code>,
                with intercept {model.intercept.toFixed(1)}.
              </li>
              <li>
                Strongest inputs (standardised coefficients):{' '}
                {strongest
                  .map((f) => `${FIELD_INFO[f.key].label} ${f.coefficient > 0 ? '+' : '−'}${Math.abs(f.coefficient).toFixed(1)}`)
                  .join(', ')}.
              </li>
              <li>
                After selection the model was trained again on all {model.trained_on} patients for
                this app. The app runs the formula in the browser and matches scikit-learn exactly.
              </li>
            </ul>
          </section>

          <section>
            <h3>Limitations</h3>
            <ul>
              <li>The model explains about {Math.round(performance.cv_r2 * 100)}% of the variation; a typical prediction is off by about {Math.round(performance.cv_mae)} points.</li>
              <li>Small dataset ({model.trained_on} patients) from a single study.</li>
              <li>Low / Normal / High are thirds of the study patients, not medical categories.</li>
              <li>Some blood measurements are strongly related to each other, so single coefficients should be read with care.</li>
              <li>The sex coding (1 = male, 2 = female) is an assumption; the dataset does not document it.</li>
              <li>For learning only, not medical advice.</li>
            </ul>
          </section>

          <section>
            <h3>Tools</h3>
            <ul>
              <li>Python, pandas, scikit-learn, SciPy and matplotlib for the analysis.</li>
              <li>React and Vite for this app. No server: the prediction runs in the browser.</li>
            </ul>
          </section>
        </div>
      </dialog>
    </>
  )
}
