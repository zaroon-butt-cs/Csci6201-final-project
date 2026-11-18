import { useEffect, useRef, useState } from 'react'
import { formatValue } from '../lib/features'

const clamp = (v, min, max) => Math.min(max, Math.max(min, v))

// Round to the step's precision so 26.1 + 0.1 gives 26.2, not 26.200000000000003
const snap = (v, step) => Number((Math.round(v / step) * step).toFixed(4))

/**
 * One numeric input: a spinner (− value +) plus a slider.
 * Holding a spinner button keeps stepping.
 */
export default function NumberField({ id, info, stats, value, effect, onChange }) {
  const { min, max, mean } = stats
  const { step } = info
  const [draft, setDraft] = useState(null) // text while the user is typing
  const hold = useRef({})

  const stopHold = () => {
    clearTimeout(hold.current.delay)
    clearInterval(hold.current.repeat)
  }
  useEffect(() => stopHold, [])
  // A button that reaches the end of the range becomes disabled and may
  // never receive "pointer up", so stop repeating here as well.
  useEffect(() => {
    if (value <= min || value >= max) stopHold()
  }, [value, min, max])

  const nudge = (direction) =>
    onChange((v) => clamp(snap(v + direction * step, step), min, max))

  const startHold = (direction) => (event) => {
    event.preventDefault()
    stopHold()
    nudge(direction)
    hold.current.delay = setTimeout(() => {
      hold.current.repeat = setInterval(() => nudge(direction), 70)
    }, 380)
  }

  const onType = (event) => {
    const text = event.target.value
    setDraft(text)
    const number = Number(text)
    // Apply straight away only when it is a valid value; otherwise wait
    // until the user leaves the box (they may still be typing).
    if (text.trim() !== '' && Number.isFinite(number) && number >= min && number <= max) {
      onChange(() => number)
    }
  }

  const onBlur = () => {
    if (draft !== null) {
      const number = Number(draft)
      if (draft.trim() !== '' && Number.isFinite(number)) onChange(() => clamp(number, min, max))
      setDraft(null)
    }
  }

  const percent = ((value - min) / (max - min)) * 100
  const meanPercent = ((mean - min) / (max - min)) * 100
  const rounded = Math.round(effect)
  const direction = rounded > 0 ? 'up' : rounded < 0 ? 'down' : 'flat'

  return (
    <div className="field">
      <div className="field-head">
        <label htmlFor={id}>{info.label}</label>
        <span
          className={`effect effect-${direction}`}
          title="How much this value moves the prediction, compared with the average patient"
        >
          {direction === 'flat' ? '±0' : `${rounded > 0 ? '+' : '−'}${Math.abs(rounded)}`}
        </span>
      </div>

      <div className="spinner">
        <button
          type="button"
          className="spin"
          aria-label={`Decrease ${info.label}`}
          disabled={value <= min}
          onPointerDown={startHold(-1)}
          onPointerUp={stopHold}
          onPointerLeave={stopHold}
          onPointerCancel={stopHold}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && nudge(-1)}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h10" /></svg>
        </button>
        <div className="spinner-value">
          <input
            id={id}
            type="number"
            inputMode="decimal"
            min={min}
            max={max}
            step={step}
            value={draft ?? formatValue(value)}
            onChange={onType}
            onBlur={onBlur}
            onKeyDown={(e) => e.key === 'Enter' && e.target.blur()}
          />
          <span className="unit">{info.unit}</span>
        </div>
        <button
          type="button"
          className="spin"
          aria-label={`Increase ${info.label}`}
          disabled={value >= max}
          onPointerDown={startHold(1)}
          onPointerUp={stopHold}
          onPointerLeave={stopHold}
          onPointerCancel={stopHold}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && nudge(1)}
        >
          <svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h10M8 3v10" /></svg>
        </button>
      </div>

      <div className="slider">
        <input
          type="range"
          aria-label={`${info.label} slider`}
          min={min}
          max={max}
          step={step}
          value={value}
          style={{ '--pct': `${percent}%` }}
          onChange={(e) => onChange(() => clamp(Number(e.target.value), min, max))}
        />
        <span className="avg-tick" style={{ left: `${meanPercent}%` }} aria-hidden="true" />
      </div>
      <div className="scale">
        <span>{formatValue(min)}</span>
        <span>average {formatValue(Number(mean.toFixed(1)))}</span>
        <span>{formatValue(max)}</span>
      </div>
    </div>
  )
}
