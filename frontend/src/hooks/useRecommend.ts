import { useState, useCallback } from 'react'
import type { RecommendResult } from '../types'

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

interface UseRecommendReturn {
  result: RecommendResult | null
  loading: boolean
  error: string | null
  recommend: (text: string) => Promise<boolean>
  reset: () => void
}

export function useRecommend(): UseRecommendReturn {
  const [result, setResult]   = useState<RecommendResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState<string | null>(null)

  const recommend = useCallback(async (text: string): Promise<boolean> => {
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API_URL}/api/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({})) as { detail?: string }
        throw new Error(body.detail ?? `Server error ${res.status}`)
      }
      const data = await res.json() as RecommendResult
      setResult(data)
      return true
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Something went wrong'
      setError(`${msg} — is the API running? (make api)`)
      return false
    } finally {
      setLoading(false)
    }
  }, [])

  const reset = useCallback((): void => {
    setResult(null)
    setError(null)
  }, [])

  return { result, loading, error, recommend, reset }
}
