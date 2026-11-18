import { BAND_LIMITS, model } from '../lib/predict'

const CX = 130
const CY = 130
const R = 104
const { min: LOW, max: HIGH } = model.target

const clamp01 = (t) => Math.min(1, Math.max(0, t))
const fraction = (score) => clamp01((score - LOW) / (HIGH - LOW))

// 0 = far left of the half circle, 1 = far right
function point(t, radius = R) {
  const angle = Math.PI * (1 - t)
  return [CX + radius * Math.cos(angle), CY - radius * Math.sin(angle)]
}

function arc(from, to, radius = R) {
  const [x1, y1] = point(from, radius)
  const [x2, y2] = point(to, radius)
  return `M ${x1.toFixed(2)} ${y1.toFixed(2)} A ${radius} ${radius} 0 0 1 ${x2.toFixed(2)} ${y2.toFixed(2)}`
}

/**
 * Half-circle gauge on the dataset's range (25 to 346).
 * The track is split into the dataset's lower / middle / upper thirds
 * (safe green, amber, critical red),
 * the soft band shows the model's typical error around the prediction.
 */
export default function Gauge({ score, error, band }) {
  const gap = 0.006
  const [t1, t2] = BAND_LIMITS.map(fraction)
  const t = fraction(score)
  const [mx, my] = point(t)

  return (
    <svg className="gauge" viewBox="0 0 260 150" role="img"
      aria-label={`Predicted progression ${Math.round(score)} on a scale from ${LOW} to ${HIGH}`}>
      <path className="zone zone-low" d={arc(0, t1 - gap)} />
      <path className="zone zone-mid" d={arc(t1 + gap, t2 - gap)} />
      <path className="zone zone-high" d={arc(t2 + gap, 1)} />

      <path className="gauge-error" d={arc(fraction(score - error), fraction(score + error), R - 17)} />

      <circle className={`gauge-marker marker-${band}`} cx={mx} cy={my} r="9" />

      <text className="gauge-end" x={CX - R} y={CY + 17} textAnchor="middle">{LOW}</text>
      <text className="gauge-end" x={CX + R} y={CY + 17} textAnchor="middle">{HIGH}</text>
    </svg>
  )
}
