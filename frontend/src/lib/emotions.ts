import type { EmotionMeta, StepMeta } from '../types'

// Vibrant palette — designed for a light background.
// Each emotion gets its own strong, readable hue so the tint shift is obvious.
export const EMOTIONS: Record<string, EmotionMeta> = {
  Happy:    { color: '#d97706', glow: 'rgba(217,119,6,0.15)',    bg: 'rgba(217,119,6,0.08)',    label: 'Happy'    },
  Sad:      { color: '#2563eb', glow: 'rgba(37,99,235,0.15)',    bg: 'rgba(37,99,235,0.07)',    label: 'Sad'      },
  Angry:    { color: '#dc2626', glow: 'rgba(220,38,38,0.15)',    bg: 'rgba(220,38,38,0.07)',    label: 'Angry'    },
  Fear:     { color: '#7c3aed', glow: 'rgba(124,58,237,0.15)',   bg: 'rgba(124,58,237,0.07)',   label: 'Fear'     },
  Disgust:  { color: '#059669', glow: 'rgba(5,150,105,0.15)',    bg: 'rgba(5,150,105,0.07)',    label: 'Disgust'  },
  Surprise: { color: '#ea580c', glow: 'rgba(234,88,12,0.15)',    bg: 'rgba(234,88,12,0.07)',    label: 'Surprise' },
}

export const STEP_META: StepMeta[] = [
  { key: 'match',       label: 'matching your mood'  },
  { key: 'bridge',      label: 'the bridge'          },
  { key: 'destination', label: 'your destination'    },
]

export const DEFAULT_ACCENT = '#7c3aed'

export function emotionColor(emotion: string): string {
  return EMOTIONS[emotion]?.color ?? DEFAULT_ACCENT
}

export function emotionBg(emotion: string): string {
  return EMOTIONS[emotion]?.bg ?? 'rgba(124,58,237,0.07)'
}
