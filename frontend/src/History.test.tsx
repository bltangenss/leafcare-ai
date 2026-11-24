import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { expect, it, vi } from 'vitest';
import { History } from './App';

it('loads older records with offset and keeps global statistics', async () => {
  const item = (id: number) => ({ id, created_at: '2026-11-22T10:00:00', filename: 'leaf.png', model: 'mobilenet', cls: 'Apple___healthy', confidence: 0.9, time_ms: 20, image_url: '/uploads/leaf.png', info: { plant: 'Алма', name: `Жапырақ ${id}`, healthy: true } });
  const stats = { total: 2, healthy: 2, affected: 0, average_confidence: 0.9, average_time_ms: 20 };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce({ ok: true, json: async () => ({ items: [item(2)], stats, limit: 1, offset: 0, has_more: true }) })
    .mockResolvedValueOnce({ ok: true, json: async () => ({ items: [item(1)], stats, limit: 1, offset: 1, has_more: false }) });
  vi.stubGlobal('fetch', fetchMock);
  render(<History onReanalyze={vi.fn()} />);
  await screen.findByText('Жапырақ 2');
  fireEvent.click(screen.getByRole('button', { name: 'Тағы көрсету' }));
  await screen.findByText('Жапырақ 1');
  expect(screen.getByText('Жапырақ 2')).toBeInTheDocument();
  expect(fetchMock.mock.calls[1][0]).toBe('/api/history?limit=50&offset=1');
  expect(screen.getByText('2 жазбаның 2 көрсетілді')).toBeInTheDocument();
  await waitFor(() => expect(screen.queryByRole('button', { name: 'Тағы көрсету' })).not.toBeInTheDocument());
});
