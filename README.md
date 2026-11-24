# LeafCare AI

Қазақ тіліндегі жапырақ суретін алдын ала жіктеу қолданбасы. React және TypeScript интерфейсі, FastAPI сервері, PyTorch CNN модельдері. Жоба `LeafCare_AI_CNN_Coursework_V1_KK.docx` есебінің функционалдық талаптары мен API келісімшартына сәйкес жаңартылды.

Нәтиже ықтимал класс ретінде беріледі. Softmax мәні модельдің дәлдігі немесе калибрленген сенімділік емес; нәтиже агрономиялық диагнозды алмастырмайды.

## Жергілікті іске қосу

Тексерілген орта: Windows, Python 3.13, Node.js 24. Python тәуелділіктерінің нұсқалары `requirements.txt`, frontend тәуелділіктері `frontend/package-lock.json` ішінде бекітілген.

```powershell
python -m pip install -r requirements.txt
cd frontend
npm ci
npm run build
cd ..
python -m app.app
```

Қолданба: http://127.0.0.1:5000. API сипаттамасы: http://127.0.0.1:5000/docs.

Windows-та frontend құрып, серверді бір командамен іске қосуға болады:

```powershell
.\start.ps1
# Дайын frontend жинағын пайдалану:
.\start.ps1 -SkipBuild
```

Бар `models/mobilenet.pt` және `models/plantcnn.pt` файлдары жеткілікті. Қолданбаны ашу үшін деректерді қайта жүктеу немесе модельді қайта оқыту қажет емес. Серверді тоқтату: Ctrl+C.

Git репозиторийінде код, дайын модельдер және бағалау метрикалары бар. Датасет, пайдаланушы жүктеген суреттер, жергілікті тарих, журналдар және эксперименттердің жұмыс каталогтары жарияланбайды. Қайта оқыту мен толық бағалауға деректер жиыны бөлек қажет; API және интерфейс тесттері оны талап етпейді.

Frontend әзірлеу режимі: сервер қосулы кезде `cd frontend; npm run dev`. Vite API сұрауларын жергілікті серверге жібереді.

## Пайдалану

1. Бір жапырақ анық көрінетін JPG немесе PNG суретін таңдаңыз. Шегі — 10 МБ.
2. Preview мен файл атауын тексеріңіз, модельді таңдаңыз.
3. Қажет болса ғана тарихқа сақтау опциясын қосып, «Жапырақты талдау» батырмасын басыңыз.
4. Ықтимал класты, top-5 нәтижені, Grad-CAM картасын, модель нұсқасын және өңдеу уақытын қараңыз.

Жаңа файл немесе басқа модель таңдалса, ескі нәтиже тазаланады. Қате немесе timeout кезінде қайта жіберуге болады. Сурет әдепкіде тұрақты сақталмайды. Пайдаланушы тарихқа сақтауды таңдағанда ғана өңделген сурет пен нәтиже жазылады; тарихтағы жазбаны жою оның суретін де жояды. Бұрынғы тарих сақталады.

## Құрылымы

```text
frontend/src/                React + TypeScript интерфейсі
frontend/dist/               браузерге берілетін жинақ
app/app.py                   FastAPI маршруттары, файл тексеруі, тарих
app/predictor.py             checkpoint тексеруі, inference, Grad-CAM
app/disease_info.py          қазақша класс анықтамалығы
ml/models.py                 CNN архитектуралары
ml/dataset.py                manifest, hash және group-aware split
ml/train.py                  қайталанатын оқыту және бағалау
ml/plots.py                  графиктер мен архитектура сұлбалары
scripts/evaluate_models.py   бар checkpoint-терді қайта бағалау
scripts/dataset_manifest.py  дерек реестрі мен duplicate аудиті
scripts/screenshots.py       тарихты өзгертпей бет скриншоттарын алу
models/                      модель салмақтары
reports/evaluation_current/  нақты қайта бағалау нәтижелері
tests/                       оқшауланған backend және ML тесттері
```

`app/templates/` және `app/static/` — бұрынғы Flask интерфейсінің бастапқы файлдары; жаңа сервер React жинағын береді. `docs/` ішіндегі бұрынғы Word құрастыру скрипттері осы тапсырмада қайта жасалмады; оларды жаңа іске асырудың есебі ретінде пайдаланбаңыз.

## API келісімшарты

| Әдіс және жол | Мақсаты |
|---|---|
| `GET /health` | Модельдің дайындығы; жарамды модель жоқ болса 503 |
| `GET /model-info` | Модельдер, SHA256 нұсқалары, кластар, preprocessing және метрикалар |
| `POST /predict` немесе `/api/predict` | Суретті жіктеу |
| `GET /api/diseases` | Қазақша анықтамалық |
| `GET /api/history` | Сақталған нәтижелер және жалпы статистика |
| `DELETE /api/history/{id}` | Жазбаны және оған тиесілі суретті жою |
| `GET /openapi.json` | Машина оқитын API схемасы |

Сурет сұрауы — `multipart/form-data`: `image`, `model`, `save_history=false`, `include_heatmap=true`.

Міндетті жауап өрістері: `request_id`, `predicted_class`, `display_name_kk`, `confidence`, `top_k`, `model_version`, `warning`. Қосымша: `top`, `info`, `heatmap`, `timings_ms`, `low_confidence`, `calibrated`, `saved_to_history`. Әрбір checkpoint нұсқасы файлдың SHA256 хэшімен байланыстырылады.

| HTTP | error_code | Себеп |
|---|---|---|
| 400 | `EMPTY_FILE` | Файл таңдалмаған немесе бос |
| 413 | `FILE_TOO_LARGE` | Файл шектен үлкен |
| 413 | `PIXEL_LIMIT_EXCEEDED` | Кескін 25 миллион пиксельден үлкен |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Кеңейтім, MIME немесе нақты формат сәйкес емес |
| 422 | `IMAGE_DECODE_FAILED` | Бүлінген, оқылмайтын кескін |
| 503 | `MODEL_UNAVAILABLE` | Модель жоқ немесе checkpoint жарамсыз |

Сервер кеңейтімге қоса MIME, decode нәтижесін және пиксель санын тексереді. Сақталатын сурет metadata-сыз қайта кодталады. Inference режимінде градиент есептелмейді; Grad-CAM үшін градиент жеке қосылып, уақытша hook міндетті түрде алынады. Бір модельге қатар келген есептеулер қорғалған.

## Эксперименттер және дәлелдер

Барлық 9 402 сурет үшін SHA256 реестрі жасалды. Бұрынғы алгоритммен қалпына келтірілген split: train 6 581, validation 1 392, test 1 429. Тексерілген файлдарда байт бойынша бірдей көшірме табылмады. Нақты өсімдік немесе түсіру сериясының идентификаторлары жоқ; perceptual duplicate тексеруі орындалмады. Сондықтан ұқсас кадрлар арқылы leakage жоқ деп тұжырым жасауға болмайды.

`reports/evaluation_current/*_evaluation.json` нәтижелері нақты бар checkpoint-терден қайта есептелді; checkpoint және manifest хэштері, per-class Precision/Recall/F1, confusion matrix және CPU latency P50/P95 қамтылған. Latency өлшеміндегі preprocessing/forward/total құрамын және Grad-CAM қосылған-қосылмағанын сол файлдан тексеріңіз. API осы метрикаларды checkpoint хэші сәйкес болғанда ғана расталған қайта бағалау ретінде көрсетеді.

PlantCNN журналы үшінші эпохада үзіледі; оның салмақтары толық оқытылған baseline деп саналмайды. Жаңа `baseline` және `regularized` архитектуралары эксперимент жүргізуге дайын, бірақ олардың серверге орнатылған салмақтары жоқ. MobileNetV2 — қолданыстағы негізгі модель. Ескі `mobilenet_results.json` және checkpoint файлдары қайта жазылмайды.

| Қолданыстағы checkpoint | Test Accuracy | Macro F1 | Test суреттері |
|---|---:|---:|---:|
| MobileNetV2 | 98.88% | 98.87% | 1 429 |
| PlantCNN | 23.79% | 21.79% | 1 429 |

Бұл — тарихи файлдық split-тің қайта бағалауы; жаңа group-aware эксперименттің немесе дала жағдайының нәтижесі емес.

Бағалауды қайта орындау:

```powershell
python scripts/evaluate_models.py
python scripts/dataset_manifest.py
python -m ml.plots
```

Бар модельдерді бағалау үшін тарихи split сақталады. Жаңа экспериментте бірдей файлдар мен белгілі source group-тарды бір бөлікке жинайтын `group_hash` саясатын қолданыңыз. Seed, split, class mapping, preprocessing, конфигурация және орта жаңа оқыту артефактілерімен бірге сақталады. Модель validation бойынша таңдалады; test параметр таңдауға қолданылмайды.

```powershell
# Жаңа толық эксперименттер; оқу уақыты компьютерге тәуелді.
python -m ml.train --model baseline --augmentation off --epochs 15 --seed 42 --run-id baseline_seed42
python -m ml.train --model regularized --augmentation on --epochs 15 --seed 42 --run-id regularized_seed42
```

Артефактілер `reports/runs/<run-id>/` ішінде сақталады: `config.json`, `environment.json`, `requirements-lock.txt`, `dataset_manifest.jsonl`, `history.csv`, `checkpoint.pt`, `metrics.json`. Бір run ID-ін қайта қолдануға жол берілмейді. Жаңа checkpoint серверге автоматты орнатылмайды. Орындалған `smoke_baseline_contract_20261122` бір эпохалық, әр класстан 10 суреті бар pipeline тексеруі ғана; ғылыми салыстыру нәтижесі ретінде қолданылмайды.

Дереккөздің бастапқы commit нұсқасы, лицензиясы және тарихи checkpoint-тердің нақты training membership хэші бұрын тіркелмеген. Ағымдағы хэш реестрі бұл ақпаратты кері қалпына келтірмейді. Сыртқы дала суреттеріндегі бағалау, calibration және OOD анықтау әлі орындалмаған; 0.5 төмен confidence белгісі — калибрленбеген эвристика.

Деректерді жаңа каталогқа жүктеу қажет болса:

```powershell
python scripts/download_dataset.py 250 --output data/plantvillage_new
```

Жүктеуші толтырылған каталогты өшірмейді және жаңа көшірме үшін дереккөздің нақты revision мәнін тіркейді.

## Тексеру

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
cd frontend
npm test
npm run build
```

Backend тесттері уақытша дерекқор мен уақытша файл каталогтарын қолданады. Олар жұмыс тарихын өзгертпейді. Есептегі WEB-01–WEB-08 сценарийлері API, predictor және frontend тесттеріне бөлінген: дұрыс сурет, қате формат, decode қатесі, үлкен файл, backend қатесі/timeout, checkpoint жоқтығы, қайта талдау және class mapping сәйкессіздігі.

Браузерлік тексеру скрипті — `scripts/browser_check.cjs`; орнатылған Playwright пен Edge қажет. `PLAYWRIGHT_MODULE` айнымалысына Playwright пакетінің абсолют жолын беруге болады. Ол нақты суретті сақтаусыз талдап, desktop/mobile беттерін, қате күйін және retry сценарийін тексереді. Нәтижелері: `reports/browser/checks.json` және PNG скриншоттары.
