import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { errorMessage, request, validateFile } from './api';
import type { Disease, HistoryData, HistoryItem, ModelInfo, Prediction } from './types';

const DISCLAIMER = 'Бұл — сурет бойынша алдын ала жіктеу. Нәтиже агрономиялық диагнозды алмастырмайды.';
const NAV = [['/', 'Жапырақты талдау'], ['/diseases', 'Анықтамалық'], ['/history', 'Талдау тарихы'], ['/about', 'Модельдер туралы']];
const percent = (value: number) => (value * 100).toFixed(1) + '%';

function Icon({ name, size = 20 }: { name: 'leaf' | 'upload' | 'arrow' | 'check' | 'info' | 'clock' | 'close' | 'image' | 'search' | 'shield' | 'trash'; size?: number }) {
  const paths: Record<string, ReactNode> = {
    leaf: <><path d="M20 3C9 1 2 7 5 14c3 7 15 3 15-11Z" /><path d="M4 21c1-7 5-11 11-14M10 11l1 5" /></>,
    upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 15v5h16v-5" /></>,
    arrow: <path d="M4 12h15m-6-6 6 6-6 6" />,
    check: <path d="m5 12 4 4L19 6" />,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10v.2" /></>,
    clock: <><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" /></>,
    close: <path d="m6 6 12 12M18 6 6 18" />,
    image: <><rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8" cy="8" r="1" /><path d="m3 17 5-5 4 4 4-7 5 8" /></>,
    search: <><circle cx="10" cy="10" r="6" /><path d="m15 15 6 6" /></>,
    shield: <><path d="m12 2 8 4v6c0 5-8 10-8 10S4 17 4 12V6Z" /><path d="m8 11 3 3 5-5" /></>,
    trash: <><path d="M4 6h16M9 6V3h6v3M6 6l1 15h10l1-15M10 10v7m4-7v7" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>;
}

function BotanicalArt() {
  return <svg className="botanical-art" viewBox="0 0 400 350" fill="none" aria-hidden="true">
    <circle cx="213" cy="165" r="127" fill="#e5ebdf" />
    <circle cx="213" cy="165" r="100" stroke="#bccab5" strokeDasharray="3 8" />
    <path d="M108 295C137 243 158 195 202 148c29-32 64-53 92-90" stroke="#3f6c4e" strokeWidth="3" />
    <path d="M163 211C100 205 89 153 101 111c40 12 74 47 62 100Z" fill="#77976b" />
    <path d="M197 163C172 107 195 61 231 45c22 36 20 82-34 118Z" fill="#446c46" />
    <path d="M237 119c-5-58 31-88 71-87 9 45-8 77-71 87Z" fill="#92aa7c" />
    <path d="M177 191c25-47 76-52 119-31-22 44-68 61-119 31Z" fill="#aac197" />
    <path d="M138 253c20-43 69-50 100-33-17 40-61 53-100 33Z" fill="#527c52" />
    <path d="m104 117 57 91m69-159-32 110m106-121-65 78m51 45-110 29m55 33-94 28" stroke="#e0e9d6" strokeOpacity=".65" strokeWidth="1.2" />
    <path d="M120 111v-19h19m152 0h19v19M120 218v19h19m152 0h19v-19" stroke="#416647" strokeWidth="2" />
    <circle cx="273" cy="268" r="24" fill="#fcfdf8" stroke="#d6dfd0" /><path d="m262 269 7 7 14-17" stroke="#3b6246" strokeWidth="2.5" />
    <circle cx="81" cy="231" r="5" fill="#b6c49a" /><circle cx="329" cy="137" r="4" fill="#b6c49a" />
  </svg>;
}

function Alert({ children, onRetry }: { children: ReactNode; onRetry?: () => void }) {
  return <div className="alert" role="alert"><Icon name="info" /><div>{children}{onRetry && <button className="text-button" onClick={onRetry}>Қайталап көру <Icon name="arrow" size={15} /></button>}</div></div>;
}

function Empty({ title, children, icon = 'leaf' }: { title: string; children: ReactNode; icon?: 'leaf' | 'clock' | 'image' }) {
  return <div className="empty-state"><span className="empty-icon"><Icon name={icon} size={32} /></span><h3>{title}</h3><p>{children}</p></div>;
}

export default function App() {
  const [path, setPath] = useState(location.pathname);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [serviceError, setServiceError] = useState('');
  const [checking, setChecking] = useState(true);
  const [reanalysis, setReanalysis] = useState<HistoryItem | null>(null);
  const refreshService = useCallback(async () => {
    setChecking(true);
    setServiceError('');
    try {
      const info = await request<ModelInfo>('/model-info', {}, 15_000);
      setModelInfo(info);
      if (!info.models.length) setServiceError('Қазір қолжетімді модель жоқ. Серверді тексеріп, қайталап көріңіз.');
    } catch (error) { setModelInfo(null); setServiceError(errorMessage(error)); }
    finally { setChecking(false); }
  }, []);
  useEffect(() => { void refreshService(); }, [refreshService]);
  useEffect(() => { const listener = () => { setPath(location.pathname); setReanalysis(null); }; window.addEventListener('popstate', listener); return () => window.removeEventListener('popstate', listener); }, []);
  function navigate(to: string) { history.pushState(null, '', to); setPath(to); setReanalysis(null); window.scrollTo({ top: 0, behavior: 'instant' }); }
  function reanalyze(item: HistoryItem) { navigate('/'); setReanalysis(item); }
  const ready = Boolean(modelInfo?.models.length) && !serviceError;
  return <>
    <a className="skip-link" href="#main">Негізгі мазмұнға өту</a>
    <header className="site-header"><div className="header-inner container">
      <a className="brand" href="/" onClick={event => { event.preventDefault(); navigate('/'); }} aria-label="LeafCare AI басты бет"><span className="brand-mark"><Icon name="leaf" size={25} /></span><span>LeafCare<span className="brand-ai">AI</span></span></a>
      <nav aria-label="Негізгі навигация">{NAV.map(([url, label]) => <a href={url} key={url} aria-current={path === url ? 'page' : undefined} onClick={event => { event.preventDefault(); navigate(url); }}>{label}</a>)}</nav>
      <span className={`service-badge ${ready ? '' : 'offline'}`}><i />{checking ? 'Қосылуда' : ready ? 'Жүйе дайын' : 'Қолжетімсіз'}</span>
    </div></header>
    <main id="main" className="container">
      {serviceError && <div className="service-alert"><Alert onRetry={() => void refreshService()}>{serviceError}</Alert></div>}
      {path === '/' ? <Analysis modelInfo={modelInfo} reanalysis={reanalysis} ready={ready} />
        : path === '/diseases' ? <Diseases />
          : path === '/history' ? <History onReanalyze={reanalyze} />
            : path === '/about' ? <About modelInfo={modelInfo} />
              : <Empty title="Бет табылмады">Мәзірден қажетті бөлімді таңдаңыз.</Empty>}
    </main>
    <footer className="site-footer container"><div className="footer-brand"><Icon name="leaf" size={19} /><span>LeafCare AI</span><span className="footer-divider">/</span><span>CNN зерттеу жобасы</span></div><span>Бір жапырақтан басталған қамқорлық.</span></footer>
  </>;
}

export function Analysis({ modelInfo, ready, reanalysis }: { modelInfo: ModelInfo | null; ready: boolean; reanalysis?: HistoryItem | null }) {
  const [selected, setSelected] = useState<File | null>(null);
  const [preview, setPreview] = useState('');
  const [model, setModel] = useState('');
  const [saveHistory, setSaveHistory] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<Prediction | null>(null);
  const [dragging, setDragging] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const resultHeading = useRef<HTMLHeadingElement>(null);
  const requestNumber = useRef(0);

  useEffect(() => { if (modelInfo && !modelInfo.models.some(item => item.id === model)) setModel(modelInfo.default_model); }, [modelInfo, model]);
  useEffect(() => {
    if (!selected) { setPreview(''); return; }
    const url = URL.createObjectURL(selected); setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [selected]);
  useEffect(() => () => { requestNumber.current += 1; controller.current?.abort(); }, []);
  function clearRequest() { requestNumber.current += 1; controller.current?.abort(); controller.current = null; setLoading(false); setResult(null); setError(''); }
  function selectFile(file?: File) {
    clearRequest();
    if (!file) { setSelected(null); return; }
    const invalid = validateFile(file);
    if (invalid) { setSelected(null); setError(invalid); if (input.current) input.current.value = ''; return; }
    setSelected(file);
  }
  useEffect(() => {
    if (!reanalysis) return;
    const aborter = new AbortController();
    const sequence = ++requestNumber.current;
    setResult(null); setError(''); setSelected(null); setModel(reanalysis.model);
    fetch(reanalysis.image_url || `/uploads/${encodeURIComponent(reanalysis.filename)}`, { signal: aborter.signal })
      .then(async response => { if (!response.ok) throw new Error('Сақталған сурет табылмады. Жаңа файл таңдаңыз.'); return response.blob(); })
      .then(blob => { if (!aborter.signal.aborted && sequence === requestNumber.current) selectFile(new File([blob], reanalysis.filename, { type: blob.type })); })
      .catch(error => { if (!aborter.signal.aborted && sequence === requestNumber.current) setError(errorMessage(error)); });
    return () => aborter.abort();
  }, [reanalysis]);

  async function analyze(event: React.FormEvent) {
    event.preventDefault();
    if (!selected || !ready || loading || controller.current) return;
    const sequence = ++requestNumber.current;
    const aborter = new AbortController(); controller.current = aborter;
    setResult(null); setError(''); setLoading(true);
    const body = new FormData(); body.append('image', selected); body.append('model', model); body.append('save_history', String(saveHistory)); body.append('include_heatmap', 'true');
    try {
      const data = await request<Prediction>('/api/predict', { method: 'POST', body, signal: aborter.signal });
      if (sequence !== requestNumber.current) return;
      if (!data.top?.length || !data.info) throw new Error('Талдау нәтижесі толық емес. Қайталап көріңіз.');
      setResult(data);
      window.setTimeout(() => resultHeading.current?.focus({ preventScroll: true }), 0);
    } catch (error) { if (sequence === requestNumber.current && !(error instanceof DOMException && error.name === 'AbortError')) setError(errorMessage(error)); }
    finally { if (sequence === requestNumber.current) { setLoading(false); controller.current = null; } }
  }

  return <>
    <section className="hero"><div className="hero-copy"><div className="eyebrow"><span />ӨСІМДІКТЕРГЕ ҚАМҚОРЛЫҚ · ЖАСАНДЫ ИНТЕЛЛЕКТ</div><h1>Жапырақ не<br /><em>айтып тұр?</em></h1><p>Бір сурет жүктеңіз. Нейрондық желі өсімдіктің ықтимал күйін бағалап, назар аударған аймағын көрсетеді.</p><div className="hero-note"><Icon name="shield" size={18} />Сурет сіздің келісіміңізбен ғана сақталады</div></div><BotanicalArt /></section>
    <div className="workflow-strip" aria-label="Талдау қадамдары"><div className={selected ? 'completed' : 'current'}><span>{selected ? <Icon name="check" size={15} /> : '01'}</span>Суретті таңдаңыз</div><i /><div className={loading || result ? 'current' : ''}><span>02</span>Талдауды бастаңыз</div><i /><div className={result ? 'current' : ''}><span>03</span>Нәтижені қараңыз</div></div>
    <section className="analysis-layout" aria-label="Жапырақты талдау">
      <form className="panel upload-panel" onSubmit={analyze}>
        <div className="panel-heading"><div><div className="section-kicker">01 / КІРІС СУРЕТІ</div><h2>Жапырақ суреті</h2></div><span className="soft-tag">JPG · PNG</span></div>
        <input ref={input} id="leaf-file" className="visually-hidden" type="file" accept="image/jpeg,image/png,.jpg,.jpeg,.png" aria-label="Жапырақ суретін таңдаңыз" onChange={event => selectFile(event.target.files?.[0])} />
        <button className={`upload-zone ${dragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`} type="button" onClick={() => input.current?.click()} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={event => { event.preventDefault(); setDragging(false); selectFile(event.dataTransfer.files[0]); }} aria-label={preview ? 'Басқа сурет таңдау' : 'Сурет жүктеу немесе файлды осында әкелу'}>
          {preview ? <><img src={preview} alt="Талдауға таңдалған жапырақ" onError={() => { clearRequest(); setError('Сурет ашылмады. Бүтін JPG немесе PNG файлын таңдаңыз.'); setSelected(null); }} /><span className="replace-photo"><Icon name="image" size={16} />Суретті ауыстыру</span></> : <><span className="upload-icon"><Icon name="upload" size={28} /></span><strong>Суретті осы жерге әкеліңіз</strong><span>немесе <b>файлды таңдаңыз</b></span><small>JPG немесе PNG · 10 МБ-қа дейін</small></>}
        </button>
        {selected && <div className="file-meta"><Icon name="image" size={17} /><span title={selected.name}>{selected.name}<small>{(selected.size / 1024 / 1024).toFixed(2)} МБ</small></span><button className="icon-button" type="button" aria-label="Суретті алып тастау" onClick={() => { selectFile(); if (input.current) input.current.value = ''; }}><Icon name="close" size={18} /></button></div>}
        <label className="field-label" htmlFor="model">Талдау моделі</label><select id="model" value={model} disabled={!modelInfo?.models.length || loading} onChange={event => { clearRequest(); setModel(event.target.value); }}>{!modelInfo?.models.length && <option value="">Модель қолжетімсіз</option>}{modelInfo?.models.map(item => <option key={item.id} value={item.id}>{item.title}</option>)}</select>
        <label className="checkbox-row"><input type="checkbox" checked={saveHistory} disabled={loading} onChange={event => setSaveHistory(event.target.checked)} /><span>Сурет пен нәтижені тарихқа сақтау<small>Белгіленбесе, сурет серверде сақталмайды.</small></span></label>
        <button type="submit" className="primary-button" disabled={!selected || loading || !ready}>{loading ? <><span className="spinner" />Талдау жүріп жатыр</> : <>Жапырақты талдау <Icon name="arrow" size={19} /></>}</button>
        {loading && <button className="cancel-button" type="button" onClick={clearRequest}>Сұрауды тоқтату</button>}
        <p className="upload-tip"><Icon name="info" size={16} />Бір жапырақты анық, табиғи жарықта түсіріңіз. Күрделі фон нәтиженің сапасына әсер етеді.</p>
      </form>
      <section className={`panel result-panel ${result ? 'has-result' : ''}`} aria-labelledby="result-title" aria-busy={loading}>
        <div className="panel-heading"><div><div className="section-kicker">02 / ТАЛДАУ НӘТИЖЕСІ</div><h2 id="result-title" ref={resultHeading} tabIndex={-1}>Ықтимал күйі</h2></div>{result && <span className="soft-tag"><Icon name="check" size={13} />Талдау аяқталды</span>}</div>
        <div aria-live="polite" aria-atomic="true">{loading && <div className="loading-state"><div className="scan-symbol"><Icon name="leaf" size={48} /><span /></div><h3>Жапырақты зерттеп жатырмыз</h3><p>Модель суретті өңдеп, ықтимал кластарды салыстырады.</p></div>}{error && <Alert>{error}</Alert>}</div>
        {!loading && !result && <Empty title={error ? 'Қайталап талдауға болады' : 'Әр жапырақтың өз тарихы бар'}>{error ? 'Файлды тексеріп, «Жапырақты талдау» батырмасын басыңыз.' : 'Жапырақ суретін таңдаңыз. Талдаудан кейін болжам, ықтимал кластар және жылу картасы осында пайда болады.'}</Empty>}
        {result && <Result data={result} image={preview} />}
        {!result && <p className="result-footnote"><Icon name="info" size={17} />{DISCLAIMER}</p>}
      </section>
    </section>
    <section className="guidance-row" aria-label="Сапалы суретке арналған кеңестер"><div><span>01</span><h3>Бір жапырақ</h3><p>Жапырақ кадрдың көп бөлігін алсын.</p></div><div><span>02</span><h3>Табиғи жарық</h3><p>Қатты көлеңке мен жарқылды болдырмаңыз.</p></div><div><span>03</span><h3>Анық көрініс</h3><p>Белгілері мен жапырақ жиегі көрінсін.</p></div></section>
  </>;
}

function Result({ data, image }: { data: Prediction; image: string }) {
  const info = data.info;
  return <div className="result-body">
    <div className="prediction-heading"><div><span className="plant-label">{info.plant}</span><h3>{data.low_confidence ? 'Сенімді жіктеу мүмкін болмады' : info.name}</h3><span className={`status-tag ${data.low_confidence ? 'uncertain' : info.healthy ? 'healthy' : 'affected'}`}>{data.low_confidence ? 'Қосымша тексеру қажет' : info.healthy ? 'Сау жапыраққа ұқсас' : 'Ауру белгісіне ұқсас'}</span></div><div className="confidence"><strong>{percent(data.confidence)}</strong><span>модель бағасы</span></div></div>
    {data.low_confidence && <p className="uncertainty-note">Ең ықтимал класс: {info.name}. Анығырақ суретпен қайталаңыз немесе маманға көрсетіңіз.</p>}
    <p className="prediction-disclaimer"><Icon name="info" size={17} />{data.warning || DISCLAIMER}</p>
    <p className="subtle-text">Softmax бағасы — болжам дәлдігінің кепілі емес. Калибрлеу және белгісіз ауруларды анықтау әзірге расталмаған.</p>
    <div className="result-images"><figure><img src={image || data.image_url} alt="Бастапқы жапырақ суреті" /><figcaption>Бастапқы сурет</figcaption></figure>{data.heatmap && <figure><img src={data.heatmap} alt="Grad-CAM модель назар аударған аймақтар" /><figcaption>Grad-CAM · назар аймағы</figcaption></figure>}</div>
    {data.heatmap && <p className="subtle-text">Жылу картасы зақымның шекарасын немесе диагноздың дұрыстығын дәлелдемейді.</p>}
    <h4>Ықтимал кластар</h4><div className="probabilities">{data.top.map((item, index) => <div key={item.cls} className="probability"><div><span><i>{String(index + 1).padStart(2, '0')}</i>{item.plant} · {item.name}</span><b>{percent(item.prob)}</b></div><progress max="1" value={item.prob} aria-label={`${item.plant}, ${item.name}: ${percent(item.prob)}`} /></div>)}</div>
    <details className="result-details"><summary>Класс туралы анықтамалық <span>+</span></summary><p>{info.symptoms}</p><h4>Жалпы күтім және алдын алу</h4><ul>{info.prevention?.map(item => <li key={item}>{item}</li>)}</ul><p className="subtle-text">Емдеу тәсілін осы болжамға ғана сүйеніп таңдамаңыз. Нақты себепті маманмен анықтаңыз.</p></details>
    <div className="result-meta"><span><Icon name="clock" size={15} />{data.time_ms.toFixed(0)} мс</span><span>{data.model_title}</span><span>{data.saved_to_history ? 'Тарихқа сақталды' : 'Сурет сақталмады'}</span></div><dl className="technical-meta"><div><dt>Модель нұсқасы</dt><dd>{data.model_version}</dd></div><div><dt>Сұрау ID</dt><dd>{data.request_id}</dd></div></dl>
  </div>;
}

function PageHeading({ kicker, title, children }: { kicker: string; title: string; children: ReactNode }) {
  return <section className="page-heading"><div className="eyebrow"><span />{kicker}</div><h1>{title}</h1><p>{children}</p></section>;
}

function Diseases() {
  const [items, setItems] = useState<Disease[]>([]);
  const [query, setQuery] = useState('');
  const [plant, setPlant] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const load = useCallback(async () => { setLoading(true); setError(''); try { setItems((await request<{ items: Disease[] }>('/api/diseases')).items); } catch (error) { setError(errorMessage(error)); } finally { setLoading(false); } }, []);
  useEffect(() => { void load(); }, [load]);
  const plants = [...new Set(items.map(item => item.plant))];
  const filtered = items.filter(item => (!plant || item.plant === plant) && `${item.plant} ${item.name} ${item.cls}`.toLowerCase().includes(query.toLowerCase()));
  return <><PageHeading kicker="ӨСІМДІКТІ ТАНЫП БІЛУ" title="Аурулар анықтамалығы">Жүйедегі өсімдік кластары, олардың белгілері және жалпы күтім туралы ақпарат.</PageHeading>
    <div className="filter-row"><label className="search-field"><Icon name="search" /><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Өсімдік немесе ауру атауы" aria-label="Анықтамалықтан іздеу" /></label><select value={plant} onChange={event => setPlant(event.target.value)} aria-label="Өсімдік түрі"><option value="">Барлық өсімдіктер</option>{plants.map(name => <option key={name}>{name}</option>)}</select><span>{filtered.length} класс</span></div>
    <p className="page-disclaimer"><Icon name="info" size={18} />Анықтамалық диагноз қоюға арналмаған. Ұқсас белгінің бірнеше себебі болуы мүмкін.</p>
    {error && <Alert onRetry={() => void load()}>{error}</Alert>}{loading && <p role="status" className="muted">Анықтамалық жүктелуде…</p>}
    <div className="disease-grid">{filtered.map(item => <details className="disease-card" key={item.cls}><summary><span className="plant-label">{item.plant}</span><span className="disease-title">{item.name}<span className="expand-sign">+</span></span><span className={`status-tag ${item.healthy ? 'healthy' : 'affected'}`}>{item.healthy ? 'Сау жапырақ' : item.kind}</span></summary><div className="disease-content"><p>{item.symptoms}</p>{!item.healthy && <p className="subtle-text">Қоздырғыш: <i>{item.pathogen}</i></p>}<h4>Алдын алу</h4><ul>{item.prevention.map(line => <li key={line}>{line}</li>)}</ul><code>{item.cls}</code></div></details>)}</div>
    {!loading && !error && filtered.length === 0 && <Empty title="Сәйкес класс табылмады" icon="image">Іздеу сөзін немесе өсімдік сүзгісін өзгертіңіз.</Empty>}
  </>;
}

export function History({ onReanalyze }: { onReanalyze: (item: HistoryItem) => void }) {
  const [data, setData] = useState<HistoryData | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [deleting, setDeleting] = useState<number | null>(null);
  const load = useCallback(async () => { setError(''); setLoading(true); try { setData(await request<HistoryData>('/api/history?limit=50&offset=0')); } catch (error) { setError(errorMessage(error)); } finally { setLoading(false); } }, []);
  useEffect(() => { void load(); }, [load]);
  async function loadMore() { if (!data || loadingMore || loading) return; setLoadingMore(true); setError(''); try { const next = await request<HistoryData>(`/api/history?limit=50&offset=${data.items.length}`); setData(previous => ({ ...next, items: [...(previous?.items ?? []), ...next.items] })); } catch (error) { setError(errorMessage(error)); } finally { setLoadingMore(false); } }
  async function remove(id: number) { if (!window.confirm('Осы нәтижені және сақталған суретін жою керек пе?')) return; setDeleting(id); try { await request(`/api/history/${id}`, { method: 'DELETE' }); await load(); } catch (error) { setError(errorMessage(error)); } finally { setDeleting(null); } }
  return <><PageHeading kicker="САҚТАЛҒАН БАҚЫЛАУЛАР" title="Талдау тарихы">Сақтауға келісім берген нәтижелер мен бұрынғы жазбалар. Қажетсіз сурет пен нәтижені кез келген уақытта жоя аласыз.</PageHeading>
    {error && <Alert onRetry={() => void load()}>{error}</Alert>}
    {data && <div className="stats-row"><Stat value={String(data.stats.total)} label="сақталған талдау" /><Stat value={String(data.stats.healthy)} label="сау класқа ұқсас" /><Stat value={String(data.stats.affected)} label="ауру класына ұқсас" /><Stat value={percent(data.stats.average_confidence || 0)} label="орташа модель бағасы" /></div>}
    {loading && <p role="status" className="muted">Тарих жүктелуде…</p>}
    {data && data.items.length === 0 && <div className="panel"><Empty title="Бақылаулар осы жерде сақталады" icon="clock">Талдау кезінде «Сурет пен нәтижені тарихқа сақтау» тармағын белгілеңіз.</Empty></div>}
    {!!data?.items.length && <div className="history-list">{data.items.map(item => <article className="history-card" key={item.id}><img src={item.image_url || `/uploads/${item.filename}`} alt={`${item.info.plant}: ${item.info.name}`} loading="lazy" /><div className="history-copy"><span className="plant-label">{item.info.plant}</span><h3>{item.info.name}</h3><p>{item.created_at.replace('T', ' ').slice(0, 19)} <span>·</span> {item.model}</p></div><div className="history-confidence"><b>{percent(item.confidence)}</b><span>модель бағасы</span></div><div className="history-actions"><button className="secondary-button" onClick={() => onReanalyze(item)}>Қайта талдау <Icon name="arrow" size={16} /></button><button className="icon-button danger" disabled={deleting !== null || loadingMore || loading} onClick={() => void remove(item.id)} aria-label={`${item.info.name} нәтижесін жою`}><Icon name="trash" size={18} /></button></div></article>)}</div>}
    {data && data.items.length > 0 && <div className="history-pagination"><span aria-live="polite">{data.stats.total} жазбаның {data.items.length} көрсетілді</span>{data.has_more && <button className="secondary-button" disabled={loadingMore || loading || deleting !== null} onClick={() => void loadMore()}>{loadingMore ? 'Жүктелуде…' : 'Тағы көрсету'} <Icon name="arrow" size={15} /></button>}</div>}
    <p className="page-disclaimer"><Icon name="info" size={18} />Тарихтағы атаулар — модель болжамдары. Олар расталған диагноз болып есептелмейді.</p>
  </>;
}

function Stat({ value, label }: { value: string; label: string }) { return <div className="stat"><strong>{value}</strong><span>{label}</span></div>; }

function About({ modelInfo }: { modelInfo: ModelInfo | null }) {
  return <><PageHeading kicker="АШЫҚ ӘДІСТЕМЕ" title="Модельдер және шектеулер">Нейрондық желілер жапырақтың түсі мен құрылымынан белгілер табады. Төменде серверге нақты жүктелген модельдер көрсетілген.</PageHeading>
    <div className="about-intro"><div><Icon name="leaf" size={30} /><h2>Суреттен ықтимал класқа</h2><p>Кескін тексеріледі, RGB форматына келтіріледі, модель өлшеміне өзгертіліп, қалыпқа келтіріледі. CNN әр класс үшін баға есептейді.</p></div><div><Icon name="shield" size={30} /><h2>Мүмкіндігін білу маңызды</h2><p>PlantVillage көбіне бақыланатын фонда түсірілген. Дала жағдайы, белгісіз ауру, бірнеше жапырақ және жарық айырмасы қате болжамға әкелуі мүмкін.</p></div></div>
    <div className="model-grid">{modelInfo?.models.map(model => {
      const metrics = model.metrics;
      const verified = model.metrics_provenance === 'verified_current_evaluation';
      const metric = (key: string) => typeof metrics?.[key] === 'number' ? metrics[key] as number : null;
      const latency = metrics?.latency_ms as { forward?: { p50?: number; p95?: number }; total?: { p50?: number; p95?: number } } | undefined;
      return <article className="panel model-card" key={model.id}><div className="panel-heading"><div><span className="section-kicker">{model.id === 'mobilenet' ? 'TRANSFER LEARNING' : 'CONVOLUTIONAL NEURAL NETWORK'}</span><h2>{model.title}</h2></div><span className="soft-tag">Жүктелген</span></div><div className="model-specs"><Stat value={String(model.classes.length)} label="танылатын класс" /><Stat value={model.input_size.join(' × ')} label="кіріс өлшемі" /></div>
        <dl className="technical-meta"><div><dt>Checkpoint нұсқасы</dt><dd>{model.version}</dd></div></dl>
        {metrics ? <><h3>{verified ? 'Осы checkpoint бойынша тексерілген нәтиже' : 'Сақталған эксперимент есебі'}</h3><p className="subtle-text">{verified ? 'Көрсеткіштер осы модель файлымен ішкі тест жиынында қайта өлшенген; checkpoint хэші сәйкес келеді.' : 'Бұл көрсеткіштер бұрынғы есептен алынған. Есеп пен осы checkpoint арасындағы байланыс хэшпен расталмаған.'} Дала жағдайындағы дәлдік өлшенбеген.</p>{metric('test_acc') !== null && metric('test_acc')! < 0.5 && <p className="uncertainty-note">Бұл модельдің ішкі тест сапасы төмен. Оны оқу тәжірибесі ретінде қарастырыңыз; негізгі талдау үшін MobileNetV2 таңдаңыз.</p>}<div className="metrics-grid">{[['test_acc', 'Test Accuracy'], ['f1_macro', 'Macro F1'], ['precision_macro', 'Macro Precision'], ['recall_macro', 'Macro Recall']].map(([key, label]) => <Stat key={key} value={metric(key) === null ? 'Өлшенбеген' : percent(metric(key)!)} label={label} />)}</div><dl className="model-facts"><div><dt>Тест суреттері</dt><dd>{metric('test_samples')?.toLocaleString('kk-KZ') ?? 'Көрсетілмеген'}</dd></div><div><dt>Оқыту дәуірлері</dt><dd>{metric('epochs') ?? 'Көрсетілмеген'}</dd></div><div><dt>Параметрлер</dt><dd>{metric('params')?.toLocaleString('kk-KZ') ?? 'Көрсетілмеген'}</dd></div><div><dt>Есептегі inference уақыты</dt><dd>{metric('infer_ms_cpu') === null ? 'Өлшенбеген' : `${metric('infer_ms_cpu')!.toFixed(1)} мс`}</dd></div></dl></> : <div className="missing-metrics"><Icon name="info" size={20} /><p>Осы модельдің сақталған сапа метрикалары жоқ. Сандық нәтиже көрсетілмейді.</p></div>}
        {verified && <div className="evaluation-evidence"><h3>Қайта тексеру материалдары</h3>{metrics?.split_policy === 'legacy' && <p className="subtle-text">Тест бөлігі бұрынғы бөлу ережесімен қайта құрылды. Бастапқы файл тізімінің хэші сақталмаған; бұл жаңа тәуелсіз немесе дала сынағы емес.</p>}<dl className="model-facts"><div><dt>Inference P50 / P95</dt><dd>{latency?.forward?.p50?.toFixed(1) ?? '—'} / {latency?.forward?.p95?.toFixed(1) ?? '—'} мс</dd></div><div><dt>Өңдеу және inference P50 / P95</dt><dd>{latency?.total?.p50?.toFixed(1) ?? '—'} / {latency?.total?.p95?.toFixed(1) ?? '—'} мс</dd></div></dl><p className="subtle-text">Grad-CAM және желілік сұрау уақыты бұл өлшемдерге кірмейді.</p><div className="evidence-links"><a href={`/reports/evaluation_current/${encodeURIComponent(model.id)}_evaluation.json`} target="_blank" rel="noreferrer">Бағалау есебі · JSON <Icon name="arrow" size={14} /></a><a href={`/reports/evaluation_current/${encodeURIComponent(model.id)}_confusion.png`} target="_blank" rel="noreferrer">Қателер матрицасын ашу <Icon name="arrow" size={14} /></a></div></div>}
      </article>;
    })}</div>
    {!modelInfo && <div className="panel"><Empty title="Модель деректері қолжетімсіз">Серверге қосылғаннан кейін модель нұсқасы мен нақты сипаттамалары осында көрсетіледі.</Empty></div>}
    <section className="method-section"><div><span className="section-kicker">НӘТИЖЕНІ ТҮСІНУ</span><h2>Баға — кепілдік емес</h2><p>Softmax мәні мен тест дәлдігі екі түрлі көрсеткіш. Модель қате жауапқа да жоғары баға бере алады. Төмен баға қосымша тексеруге себеп болады; жоғары баға оны алмастырмайды.</p></div><div><span className="section-kicker">СУРЕТ ҚҰПИЯЛЫЛЫҒЫ</span><h2>Сақтауды өзіңіз таңдайсыз</h2><p>Әдепкіде сурет тек ағымдағы талдауға қолданылады. Тарихқа сақтау бөлек таңдалады. Жазбаны жою суретті де жояды; қайта оқытуға автоматты түрде берілмейді.</p></div></section>
  </>;
}
