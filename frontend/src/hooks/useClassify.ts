import { useState, useCallback } from 'react'
import type { ClassifyResult } from '../types'

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

const MAX_FILE_MB = 30
const ALLOWED_TYPES = new Set(['audio/mpeg', 'audio/wav', 'audio/flac', 'audio/mp4', 'audio/ogg', 'audio/x-m4a'])
const ALLOWED_EXT   = new Set(['.mp3', '.wav', '.flac', '.m4a', '.ogg'])

interface UseClassifyReturn {
  result: ClassifyResult | null
  loading: boolean
  error: string | null
  classify: (file: File) => Promise<void>
  reset: () => void
}

export function useClassify(): UseClassifyReturn {
  const [result, setResult]   = useState<ClassifyResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const classify = useCallback(async (file: File): Promise<void> => {
    const ext = '.' + (file.name.split('.').pop()?.toLowerCase() ?? '')
    if (!ALLOWED_TYPES.has(file.type) && !ALLOWED_EXT.has(ext)) {
      setError(`Unsupported format. Please upload an MP3, WAV, FLAC, M4A, or OGG file.`)
      return
    }
    if (file.size > MAX_FILE_MB * 1024 * 1024) {
      setError(`File is too large. Maximum size is ${MAX_FILE_MB} MB.`)
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const form = new FormData()
      form.append('file', file)

      const res = await fetch(`${API_URL}/api/classify`, {
        method: 'POST',
        body: form,
      })

      if (!res.ok) {
        const body = await res.json().catch(() => ({})) as { detail?: string }
        throw new Error(body.detail ?? `Server error ${res.status}`)
      }

      const data = await res.json() as ClassifyResult
      setResult(data)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Something went wrong'
      setError(`${msg} — is the API running? (make api)`)
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = useCallback((): void => {
    setResult(null)
    setError(null)
  }, [])

  return { result, loading, error, classify, reset }
}
