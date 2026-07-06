import { useEffect, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'
import { Music, BarChart2 } from 'lucide-react'
import MoodInput from './components/MoodInput'
import ResultsPanel from './components/ResultsPanel'
import ClassifyTab from './components/ClassifyTab'
import { useRecommend } from './hooks/useRecommend'
import { emotionColor, emotionBg, DEFAULT_ACCENT } from './lib/emotions'

type Tab = 'recommend' | 'classify'

const TABS: { id: Tab; label: string; Icon: typeof Music }[] = [
  { id: 'recommend', label: 'Find My Path',    Icon: Music      },
  { id: 'classify',  label: 'Classify a Song', Icon: BarChart2  },
]

function TabBar({ active, onChange, accent }: { active: Tab; onChange: (t: Tab) => void; accent: string }) {
  return (
    <div className="flex gap-1 p-1 rounded-xl bg-slate-100/80 w-fit">
      {TABS.map(({ id, label, Icon }) => {
        const isActive = active === id
        return (
          <button
            key={id}
            onClick={() => onChange(id)}
            className="relative flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-sans font-medium transition-colors duration-200 focus:outline-none"
            style={{ color: isActive ? '#ffffff' : '#64748b' }}
          >
            {isActive && (
              <motion.div
                layoutId="tab-pill"
                className="absolute inset-0 rounded-lg"
                style={{ backgroundColor: accent }}
                transition={{ type: 'spring', stiffness: 400, damping: 35 }}
              />
            )}
            <span className="relative z-10 flex items-center gap-2">
              <Icon size={14} />
              {label}
            </span>
          </button>
        )
      })}
    </div>
  )
}

export default function App() {
  const [tab, setTab] = useState<Tab>('recommend')

  // Accent tracks the detected emotion from the recommend tab.
  // We lift it here so the page background + tab bar update together.
  const [accent, setAccent] = useState(DEFAULT_ACCENT)
  const [pageBg, setPageBg] = useState('transparent')

  // The recommend tab tells us when an emotion is detected so we can tint the page.
  // We pass a thin wrapper that captures the emotion before forwarding to state.
  const handleEmotionDetected = (emotion: string): void => {
    setAccent(emotionColor(emotion))
    setPageBg(emotionBg(emotion))
  }

  const handleTabChange = (t: Tab): void => {
    setTab(t)
    if (t === 'classify') {
      setAccent(DEFAULT_ACCENT)
      setPageBg('transparent')
    }
  }

  return (
    <div
      className="min-h-screen transition-colors duration-700"
      style={{
        background: `linear-gradient(160deg, #fdfcff 0%, ${pageBg === 'transparent' ? '#f3f0fb' : pageBg} 100%)`,
      }}
    >
      {/* Accent stripe */}
      <div className="h-1 w-full transition-colors duration-700" style={{ backgroundColor: accent }} />

      <main className="max-w-xl mx-auto px-5 py-12 flex flex-col gap-8">

        {/* Header */}
        <div className="flex flex-col gap-2">
          <span className="text-xs font-mono tracking-[0.25em] uppercase text-slate-400">
            Music Mood Therapy
          </span>
          <h1 className="font-display text-4xl text-slate-900 leading-tight">
            {tab === 'recommend' ? 'Tell me how you feel.' : 'What emotion is this song?'}
          </h1>
          <p className="text-base text-slate-500 font-sans mt-1 leading-relaxed">
            {tab === 'recommend'
              ? "I'll map three Nepali songs that move you from where you are toward a better place."
              : 'Upload any audio file. The MLP neural network will break down its emotion profile.'}
          </p>
        </div>

        {/* Tab bar */}
        <TabBar active={tab} onChange={handleTabChange} accent={accent} />

        {/* Tab content */}
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.22 }}
          >
            {tab === 'recommend' ? (
              <RecommendTabWrapper accent={accent} onEmotionDetected={handleEmotionDetected} />
            ) : (
              <ClassifyTab />
            )}
          </motion.div>
        </AnimatePresence>

      </main>

      <footer className="text-center pb-10">
        <p className="text-xs text-slate-300 font-mono">
          ISO Principle · Russell's Circumplex · GPT-4o-mini · Nepali MLP
        </p>
      </footer>
    </div>
  )
}

// Thin wrapper so useRecommend lives inside the tab and resets when switching.
// Calls onEmotionDetected to lift the accent color up to App.
function RecommendTabWrapper({
  accent,
  onEmotionDetected,
}: {
  accent: string
  onEmotionDetected: (emotion: string) => void
}) {
  const { result, loading, error, recommend, reset } = useRecommend()
  const resultsRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (result) {
      onEmotionDetected(result.emotion)
      resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [result, onEmotionDetected])

  const handleReset = (): void => {
    reset()
  }

  return (
    <div className="flex flex-col gap-8">
      <MoodInput onSubmit={recommend} loading={loading} accentColor={accent} />

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="text-sm font-sans text-red-600 bg-red-50 border border-red-200 rounded-xl px-4 py-3"
          >
            {error}
          </motion.div>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {result && (
          <div ref={resultsRef}>
            <ResultsPanel result={result} onReset={handleReset} />
          </div>
        )}
      </AnimatePresence>
    </div>
  )
}
