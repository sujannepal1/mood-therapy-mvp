import { useState } from 'react'
import { motion } from 'framer-motion'
import ResonanceMap from './ResonanceMap'
import { STEP_META, emotionColor } from '../lib/emotions'
import type { Song } from '../types'

// Spotify brand icon — inline SVG, no external dependency
function SpotifyIcon({ size = 14 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <path d="M12 0C5.4 0 0 5.4 0 12s5.4 12 12 12 12-5.4 12-12S18.66 0 12 0zm5.521 17.34c-.24.359-.66.48-1.021.24-2.82-1.74-6.36-2.101-10.561-1.141-.418.122-.779-.179-.899-.539-.12-.421.18-.78.54-.9 4.56-1.021 8.52-.6 11.64 1.32.42.18.479.659.301 1.02zm1.44-3.3c-.301.42-.841.6-1.262.3-3.239-1.98-8.159-2.58-11.939-1.38-.479.12-1.02-.12-1.14-.6-.12-.48.12-1.021.6-1.141C9.6 9.9 15 10.561 18.72 12.84c.361.181.54.78.241 1.2zm.12-3.36C15.24 8.4 8.82 8.16 5.16 9.301c-.6.179-1.2-.181-1.38-.721-.18-.601.18-1.2.72-1.381 4.26-1.26 11.28-1.02 15.721 1.621.539.3.719 1.02.419 1.56-.299.421-1.02.599-1.559.3z" />
    </svg>
  )
}

interface SongCardProps {
  song: Song
  index: number
  accentColor: string
}

export default function SongCard({ song, index, accentColor }: SongCardProps) {
  const [spotifyHovered, setSpotifyHovered] = useState(false)

  const step        = STEP_META[index]
  const dotColor    = emotionColor(song.emotion)
  const isDestination = index === 2
  const borderColor = isDestination ? '#d97706' : accentColor

  return (
    <motion.article
      initial={{ opacity: 0, y: 18 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay: index * 0.18, ease: [0.22, 1, 0.36, 1] }}
      className="relative bg-white rounded-xl border border-slate-100 shadow-sm overflow-hidden flex gap-4 p-5"
      style={{ borderLeftWidth: 3, borderLeftColor: borderColor }}
    >
      {/* Left: text content */}
      <div className="flex-1 min-w-0">

        {/* Step label */}
        <p
          className="text-xs font-mono tracking-widest uppercase mb-3 font-medium"
          style={{ color: borderColor }}
        >
          {step.label}
        </p>

        {/* Track title */}
        <h3 className="font-display text-xl leading-snug text-slate-900 mb-1 break-words">
          {song.track_name}
        </h3>

        {/* Artist */}
        <p className="text-sm text-slate-500 font-sans mb-4 truncate">
          {song.artists}
        </p>

        {/* Emotion pill + V/E data readout */}
        <div className="flex items-center gap-3 flex-wrap mb-4">
          <span
            className="inline-flex items-center gap-1.5 text-xs font-sans font-semibold px-2.5 py-1 rounded-full"
            style={{
              color: dotColor,
              backgroundColor: `${dotColor}15`,
              border: `1px solid ${dotColor}35`,
            }}
          >
            <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: dotColor }} />
            {song.emotion}
          </span>

          <span className="font-mono text-xs text-slate-400 tracking-wide">
            V·{song.valence.toFixed(3)}&nbsp;&nbsp;E·{song.energy.toFixed(3)}
          </span>
        </div>

        {/* Spotify button — hover managed via React state, not DOM mutation */}
        <a
          href={song.spotify_url}
          target="_blank"
          rel="noopener noreferrer"
          onMouseEnter={() => setSpotifyHovered(true)}
          onMouseLeave={() => setSpotifyHovered(false)}
          className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-sans font-semibold transition-all duration-200 active:scale-95"
          style={{
            color: '#1DB954',
            backgroundColor: spotifyHovered ? 'rgba(29,185,84,0.16)' : 'rgba(29,185,84,0.09)',
            border: '1px solid rgba(29,185,84,0.28)',
          }}
        >
          <SpotifyIcon size={13} />
          Open on Spotify
        </a>

      </div>

      {/* Right: Resonance Map */}
      <div className="flex-shrink-0 self-start mt-1">
        <ResonanceMap valence={song.valence} energy={song.energy} color={dotColor} />
      </div>
    </motion.article>
  )
}
