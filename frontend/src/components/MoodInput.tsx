import { useState, useRef, useEffect, type FormEvent, type KeyboardEvent } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ArrowRight, Loader2 } from 'lucide-react'

interface MoodInputProps {
  onSubmit: (text: string) => Promise<boolean>
  loading: boolean
  accentColor: string
}

export default function MoodInput({ onSubmit, loading, accentColor }: MoodInputProps) {
  const [text, setText] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    textareaRef.current?.focus()
  }, [])

  const submit = async (trimmed: string): Promise<void> => {
    const success = await onSubmit(trimmed)
    if (success) setText('')
  }

  const handleSubmit = (e: FormEvent): void => {
    e.preventDefault()
    const trimmed = text.trim()
    if (!trimmed || loading) return
    void submit(trimmed)
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      const trimmed = text.trim()
      if (trimmed && !loading) void submit(trimmed)
    }
  }

  const canSubmit = text.trim().length > 0 && !loading

  return (
    <form onSubmit={handleSubmit} className="w-full flex flex-col gap-4">
      <div
        className="rounded-xl transition-all duration-300"
        style={{
          boxShadow: text.trim()
            ? `0 0 0 2px ${accentColor}40, 0 1px 3px rgba(0,0,0,0.06)`
            : '0 0 0 1.5px #e2e0ea, 0 1px 3px rgba(0,0,0,0.04)',
        }}
      >
        <textarea
          ref={textareaRef}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="I've been feeling…"
          rows={4}
          disabled={loading}
          className="
            w-full resize-none rounded-xl
            bg-white text-slate-800 font-sans text-base
            placeholder:text-slate-300
            px-5 py-4
            focus:outline-none
            disabled:opacity-50
            leading-relaxed
          "
          style={{ caretColor: accentColor }}
        />
      </div>

      <div className="flex items-center justify-between">
        <span className="text-xs text-slate-400 font-mono">
          Enter to submit · Shift+Enter for new line
        </span>

        <button
          type="submit"
          disabled={!canSubmit}
          className="
            inline-flex items-center gap-2
            px-5 py-2.5
            rounded-xl
            text-sm font-sans font-semibold
            transition-all duration-200
            disabled:opacity-30 disabled:cursor-not-allowed
            active:scale-95
          "
          style={{
            backgroundColor: canSubmit ? accentColor : '#e2e0ea',
            color: canSubmit ? '#ffffff' : '#9e9ab5',
          }}
        >
          <AnimatePresence mode="wait">
            {loading ? (
              <motion.span
                key="loading"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2"
              >
                <Loader2 size={14} className="animate-spin" />
                Reading your mood…
              </motion.span>
            ) : (
              <motion.span
                key="idle"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="flex items-center gap-2"
              >
                Find my path
                <ArrowRight size={14} />
              </motion.span>
            )}
          </AnimatePresence>
        </button>
      </div>
    </form>
  )
}
