import { FIELD_INFO } from '../lib/features'

/**
 * Shows how far each input moves the prediction away from the average
 * patient. Bars to the right raise the score, bars to the left lower it.
 */
export default function Drivers({ contributions }) {
  const sorted = [...contributions].sort((a, b) => Math.abs(b.effect) - Math.abs(a.effect))
  const largest = Math.max(1, ...sorted.map((c) => Math.abs(c.effect)))

  return (
    <ul className="drivers">
      {sorted.map(({ key, effect }) => {
        const width = (Math.abs(effect) / largest) * 50
        const rounded = Math.round(effect)
        const up = effect > 0
        return (
          <li key={key}>
            <span className="driver-name">{FIELD_INFO[key].label}</span>
            <span className="driver-track" aria-hidden="true">
              <span
                className={`driver-bar ${up ? 'is-up' : 'is-down'}`}
                style={{ width: `${width}%`, [up ? 'left' : 'right']: '50%' }}
              />
            </span>
            <span className="driver-value">
              {rounded === 0 ? '0' : `${rounded > 0 ? '+' : '−'}${Math.abs(rounded)}`}
            </span>
          </li>
        )
      })}
    </ul>
  )
}
