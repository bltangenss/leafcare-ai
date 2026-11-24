import { describe, expect, it, vi } from 'vitest';
import { MAX_FILE_SIZE, request, validateFile } from './api';

describe('image validation', () => {
  it('rejects MIME/extension disagreement and empty images', () => {
    expect(validateFile(new File(['image'], 'leaf.jpg', { type: 'text/plain' }))).toMatch(/JPG/);
    expect(validateFile(new File(['image'], 'leaf.webp', { type: 'image/jpeg' }))).toMatch(/JPG/);
    expect(validateFile(new File([], 'leaf.png', { type: 'image/png' }))).toMatch(/бос/);
    expect(validateFile(new File(['image'], 'leaf.PNG', { type: 'image/png' }))).toBeNull();
  });
  it('rejects images over the exact server size limit', () => {
    const file = new File([new Uint8Array(MAX_FILE_SIZE + 1)], 'big.png', { type: 'image/png' });
    expect(validateFile(file)).toMatch(/10 МБ/);
  });
});

describe('API failures', () => {
  it('preserves meaningful server validation errors', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 422, json: async () => ({ error: 'Кескін бүлінген.' }) }));
    await expect(request('/api/predict')).rejects.toThrow('Кескін бүлінген.');
  });
  it('reports disconnected backend in Kazakh', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('Failed to fetch')));
    await expect(request('/health')).rejects.toThrow('Сервермен байланыс жоқ');
  });
  it('aborts a stalled server and reports a retryable timeout', async () => {
    vi.useFakeTimers();
    let signal: AbortSignal | undefined;
    vi.stubGlobal('fetch', vi.fn((_url: string, options: RequestInit) => new Promise((_resolve, reject) => {
      signal = options.signal as AbortSignal;
      signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    })));
    const pending = expect(request('/api/predict', {}, 100)).rejects.toThrow('күту уақыты аяқталды');
    await vi.advanceTimersByTimeAsync(100);
    await pending;
    expect(signal?.aborted).toBe(true);
  });
});
