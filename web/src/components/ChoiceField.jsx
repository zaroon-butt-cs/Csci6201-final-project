/** A two-option input shown as a segmented switch (used for sex: code 1 / 2). */
export default function ChoiceField({ id, info, value, effect, onChange }) {
  const rounded = Math.round(effect)
  const direction = rounded > 0 ? 'up' : rounded < 0 ? 'down' : 'flat'

  return (
    <div className="field">
      <div className="field-head">
        <span id={`${id}-label`} className="field-label">{info.label}</span>
        <span
          className={`effect effect-${direction}`}
          title="How much this value moves the prediction, compared with the average patient"
        >
          {direction === 'flat' ? '±0' : `${rounded > 0 ? '+' : '−'}${Math.abs(rounded)}`}
        </span>
      </div>
      <div className="segmented" role="radiogroup" aria-labelledby={`${id}-label`}>
        {info.options.map((option) => (
          <label key={option.value} className={value === option.value ? 'is-on' : ''}>
            <input
              type="radio"
              name={id}
              checked={value === option.value}
              onChange={() => onChange(() => option.value)}
            />
            {option.label}
          </label>
        ))}
      </div>
      <p className="field-note">{info.note}</p>
    </div>
  )
}
