import { useEffect, useRef, useState } from 'react'

const DURATION = 380 // ms

// Smoothly moves a displayed number toward `target` whenever it changes.
// Users who ask for reduced motion get the new value immediately.
export function useAnimatedNumber(target) {
  const [value, setValue] = useState(target)
  const current = useRef(target)

  useEffect(() => {
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    const duration = reduced ? 0 : DURATION

    const from = current.current
    const start = performance.now()
    let frame

    const tick = (now) => {
      const t = duration ? Math.min((now - start) / duration, 1) : 1
      const eased = 1 - (1 - t) ** 3
      current.current = from + (target - from) * eased
      setValue(current.current)
      if (t < 1) frame = requestAnimationFrame(tick)
    }
    frame = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(frame)
  }, [target])

  return value
}
