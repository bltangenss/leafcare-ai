export const MAX_FILE_SIZE = 10 * 1024 * 1024;
export const REQUEST_TIMEOUT = 45_000;

export function validateFile(file: File): string | null {
  if (!file.size) return 'Файл бос. Басқа суретті таңдаңыз.';
  if (file.size > MAX_FILE_SIZE) return 'Файл көлемі 10 МБ-тан аспауы керек.';
  if (!/\.(jpe?g|png)$/i.test(file.name) || !['image/jpeg', 'image/png'].includes(file.type)) return 'Тек JPG немесе PNG суретін таңдаңыз.';
  return null;
}

export async function request<T>(url: string, options: RequestInit = {}, timeoutMs = REQUEST_TIMEOUT): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const abort = () => controller.abort();
  if (options.signal?.aborted) controller.abort();
  options.signal?.addEventListener('abort', abort, { once: true });
  const timer = window.setTimeout(() => { timedOut = true; controller.abort(); }, timeoutMs);
  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    const data = await response.json().catch(() => null);
    if (!response.ok) throw new Error(data?.error || (response.status === 503 ? 'Модель уақытша қолжетімсіз. Кейін қайталап көріңіз.' : 'Сервер сұрауды орындай алмады. Қайталап көріңіз.'));
    if (data === null) throw new Error('Сервер жауабы оқылмады. Қайталап көріңіз.');
    return data as T;
  } catch (error) {
    if (timedOut) throw new Error('Серверден жауап күту уақыты аяқталды. Қайта талдап көріңіз.');
    if (controller.signal.aborted) throw new DOMException('Aborted', 'AbortError');
    if (error instanceof TypeError) throw new Error('Сервермен байланыс жоқ. Қосылымды тексеріп, қайталап көріңіз.');
    throw error;
  } finally {
    window.clearTimeout(timer);
    options.signal?.removeEventListener('abort', abort);
  }
}

export const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'Күтпеген қате пайда болды.';
