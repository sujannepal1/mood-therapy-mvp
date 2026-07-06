import { motion } from 'framer-motion'
import { RotateCcw } from 'lucide-react'
import SongCard from './SongCard'
import { emotionColor, emotionBg } from '../lib/emotions'
import type { RecommendResult } from '../types'

interface ResultsPanelProps {
  result: RecommendResult
  onReset: () => void
}

export default function ResultsPanel({ result, onReset }: ResultsPanelProps) {
  const { emotion, songs } = result
  const accent = emotionColor(emotion)
  const bg = emotionBg(emotion)

  return (
    <motion.section
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.35 }}
      className="w-full flex flex-col gap-5"
    >
      {/* ISO description row — emotion badge sits on the right */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.05 }}
        className="flex items-center justify-between gap-4"
      >
        <p className="text-xs text-slate-400 font-sans leading-relaxed">
          ISO Principle — three songs guide you from where you are toward where you want to be.
          Listen in order.
        </p>

        {/* Detected emotion badge — right-aligned */}
        <span
          className="flex-shrink-0 inline-flex items-center gap-1.5 text-xs font-sans font-bold px-3 py-1.5 rounded-full whitespace-nowrap"
          style={{ color: accent, backgroundColor: bg, border: `1px solid ${accent}35` }}
        >
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: accent }} />
          {emotion}
        </span>
      </motion.div>

      {/* Divider */}
      <div className="h-px bg-slate-100" />

      {/* Song cards */}
      <div className="flex flex-col gap-3">
        {songs.map((song, i) => (
          <SongCard key={song.track_id} song={song} index={i} accentColor={accent} />
        ))}
      </div>

      {/* Reset */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.4, delay: 0.65 }}
        className="flex justify-center pt-2"
      >
        <button
          onClick={onReset}
          className="inline-flex items-center gap-2 text-sm font-sans font-medium text-slate-400 hover:text-slate-700 transition-colors duration-200 group"
        >
          <RotateCcw
            size={13}
            className="group-hover:rotate-180 transition-transform duration-500"
          />
          Try a different mood
        </button>
      </motion.div>
    </motion.section>
  )
}
