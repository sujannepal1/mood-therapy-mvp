import { useId } from 'react'

interface ResonanceMapProps {
  valence: number
  energy: number
  color: string
}

/**
 * ResonanceMap — signature element.
 *
 * 72×72 SVG showing where a song sits on Russell's Circumplex Model.
 * X = valence (left → right), Y = energy (bottom → top, SVG Y is flipped).
 *
 * Across the three song cards you can watch the dot migrate from the
 * user's current emotional state toward the Happy quadrant (top-right).
 */
export default function ResonanceMap({ valence, energy, color }: ResonanceMapProps) {
  const SIZE  = 72
  const PAD   = 8
  const INNER = SIZE - PAD * 2

  const v = Math.max(0, Math.min(1, valence))
  const e = Math.max(0, Math.min(1, energy))

  const cx = PAD + v * INNER
  const cy = PAD + (1 - e) * INNER  // flip Y: high energy = top

  // useId gives each SVG instance a guaranteed-unique filter ID.
  // Without this, two songs landing at the same grid position would share
  // the same <filter> element and the glow effect would silently break.
  const uid      = useId()
  const filterId = `glow-${uid.replace(/:/g, '')}`

  return (
    <svg
      width={SIZE}
      height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      role="img"
      aria-label={`Resonance map: valence ${valence.toFixed(3)}, energy ${energy.toFixed(3)}`}
      style={{ flexShrink: 0 }}
    >
      <defs>
        <filter id={filterId} x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="3.5" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Background */}
      <rect width={SIZE} height={SIZE} rx="8" fill="#f3f0fb" />

      {/* Grid lines */}
      {[0.25, 0.5, 0.75].map((t) => {
        const gx = PAD + t * INNER
        const gy = PAD + t * INNER
        return (
          <g key={t}>
            <line x1={gx} y1={PAD} x2={gx} y2={SIZE - PAD} stroke="#1c1033" strokeOpacity="0.07" strokeWidth="0.5" />
            <line x1={PAD} y1={gy} x2={SIZE - PAD} y2={gy} stroke="#1c1033" strokeOpacity="0.07" strokeWidth="0.5" />
          </g>
        )
      })}

      {/* Plot area border */}
      <rect
        x={PAD} y={PAD} width={INNER} height={INNER} rx="3"
        fill="none" stroke="#1c1033" strokeOpacity="0.1" strokeWidth="0.5"
      />

      {/* Axis labels */}
      <text x={PAD + 1} y={PAD + 7} fill="#6e6b8a" fontSize="5" fontFamily="JetBrains Mono">E↑</text>
      <text x={SIZE - PAD - 9} y={SIZE - PAD - 2} fill="#6e6b8a" fontSize="5" fontFamily="JetBrains Mono">V→</text>

      {/* Happy target zone — faint warm indicator top-right */}
      <circle cx={PAD + 0.8 * INNER} cy={PAD + 0.2 * INNER} r="4" fill="#d97706" fillOpacity="0.15" />

      {/* Song position dot with glow */}
      <circle cx={cx} cy={cy} r="5.5" fill={color} fillOpacity="0.2" filter={`url(#${filterId})`} />
      <circle cx={cx} cy={cy} r="3.5" fill={color} />
    </svg>
  )
}
