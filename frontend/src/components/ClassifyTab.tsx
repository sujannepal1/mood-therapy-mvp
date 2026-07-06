import { useRef, useState, type DragEvent, type ChangeEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Loader2, RotateCcw, Music } from 'lucide-react'
import { useClassify } from '../hooks/useClassify'
import { EMOTIONS, DEFAULT_ACCENT } from '../lib/emotions'

function EmotionBars({ predictions }: { predictions: Record<string, number> }) {
  const sorted = Object.entries(predictions).sort((a, b) => b[1] - a[1])
  const topEmotion = sorted[0][0]

  return (
    <div className="flex flex-col gap-3">
      {sorted.map(([emotion, prob], i) => {
        const meta    = EMOTIONS[emotion]
        const color   = meta?.color ?? DEFAULT_ACCENT
        const pct     = Math.round(prob * 100)
        const isTop   = emotion === topEmotion

        return (
          <motion.div
            key={emotion}
            initial={{ opacity: 0, x: -12 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.35, delay: i * 0.07 }}
            className="flex flex-col gap-1.5"
          >
            <div className="flex items-center justify-between">
              <span
                className="text-sm font-sans font-semibold"
                style={{ color: isTop ? color : '#64748b' }}
              >
                {emotion}
              </span>
              <span
                className="text-sm font-mono font-bold"
                style={{ color: isTop ? color : '#94a3b8' }}
              >
                {pct}%
              </span>
            </div>
            <div className="relative h-2 rounded-full bg-slate-100 overflow-hidden">
              <motion.div
                className="absolute inset-y-0 left-0 rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${pct}%` }}
                transition={{ duration: 0.6, delay: i * 0.07 + 0.15, ease: [0.22, 1, 0.36, 1] }}
                style={{ backgroundColor: color, opacity: isTop ? 1 : 0.45 }}
              />
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}

export default function ClassifyTab() {
  const { result, loading, error, classify, reset } = useClassify()
  const [dragActive, setDragActive] = useState(false)
  const [fileName, setFileName]     = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = (file: File | null | undefined): void => {
    if (!file) return
    setFileName(file.name)
    void classify(file)
  }

  const onDrop = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault()
    setDragActive(false)
    handleFile(e.dataTransfer.files[0])
  }

  const onDragOver = (e: DragEvent<HTMLDivElement>): void => {
    e.preventDefault()
    setDragActive(true)
  }

  const onDragLeave = (): void => setDragActive(false)

  const onChange = (e: ChangeEvent<HTMLInputElement>): void => {
    handleFile(e.target.files?.[0])
    e.target.value = ''
  }

  const handleReset = (): void => {
    reset()
    setFileName(null)
  }

  const topEmotion  = result?.top_emotion ?? null
  const topMeta     = topEmotion ? EMOTIONS[topEmotion] : null
  const accentColor = topMeta?.color ?? DEFAULT_ACCENT

  return (
    <div className="flex flex-col gap-6">

      {/* Upload zone — hidden when result is showing */}
      <AnimatePresence>
        {!result && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3 }}
          >
            <div
              onClick={() => !loading && inputRef.current?.click()}
              onDrop={onDrop}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              className="relative flex flex-col items-center justify-center gap-4 rounded-xl border-2 border-dashed px-8 py-12 text-center transition-all duration-200 cursor-pointer select-none"
              style={{
                borderColor: dragActive ? accentColor : '#d1c4e9',
                backgroundColor: dragActive ? `${accentColor}08` : '#fdfcff',
              }}
            >
              <input
                ref={inputRef}
                type="file"
                accept=".mp3,.wav,.flac,.m4a,.ogg,audio/*"
                className="hidden"
                onChange={onChange}
                disabled={loading}
              />

              <div
                className="flex items-center justify-center w-12 h-12 rounded-full transition-colors duration-200"
                style={{ backgroundColor: dragActive ? `${accentColor}18` : '#f3f0fb' }}
              >
                {loading ? (
                  <Loader2 size={22} className="animate-spin text-slate-400" />
                ) : (
                  <Upload size={22} style={{ color: dragActive ? accentColor : '#7c3aed' }} />
                )}
              </div>

              <div className="flex flex-col gap-1">
                {loading ? (
                  <>
                    <p className="text-sm font-sans font-semibold text-slate-700">
                      Analysing audio features…
                    </p>
                    <p className="text-xs text-slate-400 font-sans">
                      Running librosa + MLP neural network
                    </p>
                  </>
                ) : (
                  <>
                    <p className="text-sm font-sans font-semibold text-slate-700">
                      {fileName ? fileName : 'Drop your song here, or click to browse'}
                    </p>
                    <p className="text-xs text-slate-400 font-sans">
                      MP3 · WAV · FLAC · M4A · OGG · up to 30 MB
                    </p>
                  </>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error */}
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

      {/* Result */}
      <AnimatePresence>
        {result && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4 }}
            className="flex flex-col gap-6"
          >
            {/* Top emotion hero */}
            <div
              className="flex flex-col items-start gap-3 rounded-xl p-5"
              style={{ backgroundColor: topMeta ? `${topMeta.color}0e` : '#f3f0fb' }}
            >
              <div className="flex items-center gap-3">
                <div
                  className="flex items-center justify-center w-10 h-10 rounded-full"
                  style={{ backgroundColor: accentColor }}
                >
                  <Music size={18} color="white" />
                </div>
                <div>
                  <p className="text-xs font-mono tracking-widest uppercase text-slate-400">
                    {fileName ?? 'Your song'}
                  </p>
                  <p className="text-lg font-display text-slate-900 leading-tight">
                    This song feels{' '}
                    <span style={{ color: accentColor }} className="font-semibold">
                      {topEmotion}
                    </span>
                  </p>
                </div>
              </div>
              <p className="text-xs text-slate-500 font-sans leading-relaxed">
                MLP neural network classified this from 7 audio features extracted with librosa —
                valence, energy, tempo, mode, loudness, acousticness, danceability.
              </p>
            </div>

            <div className="h-px bg-slate-100" />

            {/* Probability bars */}
            <div>
              <p className="text-xs font-mono tracking-widest uppercase text-slate-400 mb-4">
                Emotion breakdown
              </p>
              <EmotionBars predictions={result.predictions} />
            </div>

            {/* Reset */}
            <div className="flex justify-center pt-1">
              <button
                onClick={handleReset}
                className="inline-flex items-center gap-2 text-sm font-sans font-medium text-slate-400 hover:text-slate-700 transition-colors duration-200 group"
              >
                <RotateCcw size={13} className="group-hover:rotate-180 transition-transform duration-500" />
                Classify another song
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
