import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { Analysis } from './App';
import type { ModelInfo, Prediction } from './types';

const info: ModelInfo = { default_model: 'mobilenet', models: [
  { id: 'mobilenet', title: 'MobileNetV2', version: 'sha256:test', input_size: [192, 192], classes: [], metrics: null },
  { id: 'plantcnn', title: 'PlantCNN', version: 'sha256:other', input_size: [128, 128], classes: [], metrics: null },
], warning: '', retention: {} };
const prediction: Prediction = {
  request_id: 'request-example', predicted_class: 'Apple___healthy', display_name_kk: 'Сау жапырақ', confidence: 0.91,
  model_version: 'sha256:test', warning: 'Бұл — алдын ала жіктеу.', model: 'mobilenet', model_title: 'MobileNetV2', time_ms: 22,
  heatmap: null, low_confidence: false, calibrated: false, saved_to_history: false,
  top: [{ cls: 'Apple___healthy', prob: 0.91, plant: 'Алма', name: 'Сау жапырақ', healthy: true }],
  info: { cls: 'Apple___healthy', plant: 'Алма', name: 'Сау жапырақ', healthy: true, kind: 'Сау', severity: 0, severity_text: '', pathogen: '', symptoms: 'Сыртқы белгі жоқ.', prevention: [], treatment: [] },
};
const file = (name = 'leaf.png') => new File(['png bytes'], name, { type: 'image/png' });
const choose = (image = file()) => fireEvent.change(screen.getByLabelText('Жапырақ суретін таңдаңыз'), { target: { files: [image] } });

beforeEach(() => {
  vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: vi.fn(() => 'blob:preview'), revokeObjectURL: vi.fn() }));
});

describe('analysis interaction', () => {
  it('requires an explicit submit and defaults to no persistent history', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => prediction });
    vi.stubGlobal('fetch', fetchMock);
    render(<Analysis modelInfo={info} ready />);
    expect(screen.getByRole('button', { name: /^Жапырақты талдау/ })).toBeDisabled();
    choose();
    expect(fetchMock).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole('button', { name: /^Жапырақты талдау/ }));
    await screen.findByText('Сау жапыраққа ұқсас');
    const body = fetchMock.mock.calls[0][1].body as FormData;
    expect(body.get('save_history')).toBe('false');
    expect(body.get('model')).toBe('mobilenet');
    expect(screen.getByText('request-example')).toBeInTheDocument();
    expect(screen.getByText('Сурет сақталмады')).toBeInTheDocument();
  });
  it('clears stale predictions when a new photo or model is chosen', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => prediction }));
    render(<Analysis modelInfo={info} ready />);
    choose(); fireEvent.click(screen.getByRole('button', { name: /^Жапырақты талдау/ }));
    await screen.findByText('request-example');
    choose(file('second.png'));
    expect(screen.queryByText('request-example')).not.toBeInTheDocument();
    expect(screen.getByText('second.png')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /^Жапырақты талдау/ }));
    await screen.findByText('request-example');
    fireEvent.change(screen.getByLabelText('Талдау моделі'), { target: { value: 'plantcnn' } });
    expect(screen.queryByText('request-example')).not.toBeInTheDocument();
  });
  it('clears a previously valid photo after invalid replacement', () => {
    render(<Analysis modelInfo={info} ready />);
    choose();
    choose(new File(['text'], 'file.txt', { type: 'text/plain' }));
    expect(screen.getByRole('alert')).toHaveTextContent('Тек JPG немесе PNG');
    expect(screen.getByRole('button', { name: /^Жапырақты талдау/ })).toBeDisabled();
    expect(screen.queryByAltText('Талдауға таңдалған жапырақ')).not.toBeInTheDocument();
  });
  it('shows uncertain status without presenting a diagnosis', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ...prediction, confidence: 0.3, low_confidence: true }) }));
    render(<Analysis modelInfo={info} ready />);
    choose(); fireEvent.click(screen.getByRole('button', { name: /^Жапырақты талдау/ }));
    expect(await screen.findByText('Сенімді жіктеу мүмкін болмады')).toBeInTheDocument();
    expect(screen.getByText('Қосымша тексеру қажет')).toBeInTheDocument();
  });
  it('recovers from a server failure and permits retry', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503, json: async () => ({ error: 'Модель қолжетімсіз.' }) }));
    render(<Analysis modelInfo={info} ready />);
    choose(); fireEvent.click(screen.getByRole('button', { name: /^Жапырақты талдау/ }));
    expect(await screen.findByRole('alert')).toHaveTextContent('Модель қолжетімсіз.');
    await waitFor(() => expect(screen.getByRole('button', { name: /^Жапырақты талдау/ })).toBeEnabled());
  });
});
