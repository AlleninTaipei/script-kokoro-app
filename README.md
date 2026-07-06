# script-kokoro-app

本 repo 是在 [hexgrad/kokoro](https://github.com/hexgrad/kokoro) 之上, 把 `kokoro` 當作一般 pip 依賴, 進行相容性測試與劇本多角色語音合成的應用程式, 並提供一個 Web UI(`web_app/`)讓你透過瀏覽器完成完整流程。

> 注意 : 以下檔案不納入版本控制, 需在本機自行產生或下載 :
> - `web_app/tmp/*.wav` : Web UI 合成的暫存音訊
> - `*.wav` : 合成輸出的音訊檔案(執行腳本後於根目錄或當前目錄產生)
> - `kokoro.onnx` : 由 `test_onnx.py` / `test_onnx_pc.py` 從 PyTorch 模型匯出(約 311 MB)

## 環境設定

Python 3.13 與 kokoro 不相容(`misaki` 依賴限制), 需用 Python 3.12 虛擬環境 :

```bash
# 在 repo 根目錄執行(mac / linux)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

Windows :

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Intel Mac(macOS 12, x86_64)需要額外套用版本釘住(`torch` 最高支援 2.2.2, `numpy` 需降至 `<2`, `transformers` 需降至 `<5`) :

```bash
uv pip install --python .venv/bin/python -r requirements.txt -r requirements-mac-legacy.txt
```

Windows 或其他有完整 torch wheel 的平台不需要這個檔案, 直接裝 `requirements.txt` 的最新版即可。

---

## Web UI(推薦)

`web_app/` 提供完整的瀏覽器介面, 整合 LLM 自動配音與即時串流進度。

### 啟動

API Key 為選用設定, 不設定也能完整使用多角色配音(改為手動指定聲音), 詳見下方「功能流程」。若要啟用 AI 自動配音, 複製 `.env.example` 為 `.env`, 依需求填入 Anthropic / OpenAI / Google 三選一的 API Key, 並在 `DEFAULT_PROVIDER` 指定要用哪一家 :

```bash
cp .env.example .env
# 編輯 .env : 設定 DEFAULT_PROVIDER(anthropic / openai / google), 並填入對應的 API Key
```

啟動 server :

```bash
# mac / linux
.venv/bin/python -m uvicorn web_app.app:app --port 8000

# Windows
.venv/Scripts/python.exe -m uvicorn web_app.app:app --port 8000
```

瀏覽器開 `http://localhost:8000`

### 功能流程

1. 上傳劇本(.txt, 支援中英文)
2. 角色配音 : 頁面會自動偵測後端是否已設定 API Key, 決定「使用 AI 自動配音」勾選框的預設狀態。勾選時, 會呼叫 `.env` 中 `DEFAULT_PROVIDER` 指定的雲端 LLM, 根據台詞語氣, 性別, 語言為每個角色選出最合適的 Kokoro 聲音; 取消勾選, 或呼叫 LLM 失敗(例如未設定 API Key), 則會改用輪詢方式先預設分配聲音。兩種情況下, 角色配音表格都可以手動調整
3. 開始合成 : SSE 串流, 逐句顯示進度, 角色, RTF 數值
4. 播放與下載 : HTML5 播放器直接試聽, 或下載完整 WAV / MP3(MP3 需系統安裝 `ffmpeg`)

右上角可切換推論引擎 : ONNX(快, 約 4x)或 PyTorch(慢, 無需額外步驟)。

### 可用聲音

| 語言 | 代碼 | 女聲 | 男聲 |
|---|---|---|---|
| American English | `a` | af_heart, af_bella, af_jessica, af_nicole, af_sarah | am_adam, am_michael, am_echo, am_eric, am_liam |
| British English | `b` | bf_emma | bm_george, bm_lewis |
| Mandarin Chinese | `z` | zf_xiaobei, zf_xiaoni, zf_xiaoxiao, zf_xiaoyi | zm_yunjian, zm_yunxi, zm_yunxia, zm_yunyang |

---

## CLI 腳本說明

### `test_mac.py` : 環境與 PyTorch 推論測試

確認這台 Mac 能否正常執行 Kokoro TTS(PyTorch 路徑)。

```bash
.venv/bin/python test_mac.py
```

### `test_pc.py` : Windows 環境與 PyTorch 推論測試

`test_mac.py` 的 Windows 對應版本。播放音檔改用內建的 `winsound`, 不依賴 `afplay`。

```bash
.venv/Scripts/python.exe test_pc.py
```

> Windows 上沒有 Intel Mac 缺 wheel 的問題, 直接安裝最新版 `torch` / `numpy` / `transformers` 即可。

### `test_onnx.py` : ONNX 推論測試(Mac)

將 PyTorch 模型 export 成 `kokoro.onnx`, 再用 `onnxruntime` 跑推論, 驗證 ONNX 路徑。第一次執行會在根目錄產生 `kokoro.onnx`(約 311 MB, 已排除於版本控制), 之後重複執行會跳過 export 步驟。

```bash
.venv/bin/python test_onnx.py
```

### `test_onnx_pc.py` : ONNX 推論測試(Windows)

`test_onnx.py` 的 Windows 對應版本, 流程相同(匯出, onnxruntime 推論, 播放), 播放改用 `winsound`。

```bash
.venv/Scripts/python.exe test_onnx_pc.py
```

> torch >= 2.6 預設改用新的 dynamo-based ONNX exporter, 對這個模型的動態控制流會匯出失敗, 已在 `torch.onnx.export()` 加上 `dynamo=False` 強制走舊版 TorchScript-based exporter。

### `script_to_audio.py` : 劇本轉語音(PyTorch 版)

多角色劇本批次合成, 每個角色自動分配不同的 Kokoro 音色。

```bash
.venv/bin/python script_to_audio.py example_story.txt

# 手動指定角色音色
.venv/bin/python script_to_audio.py example_story.txt --voices "Alice=af_heart,Bob=am_adam"

# 列出所有可用音色
.venv/bin/python script_to_audio.py --list-voices
```

支援兩種劇本格式 :

```
# 格式 A(對話格式)
Alice: Hello, how are you?
Bob: I'm doing great!

# 格式 B(標籤格式, 與 TTS-Story 相容)
[alice]Hello, how are you?[/alice]
[bob]I'm doing great![/bob]
```

### `script_to_audio_onnx.py` : 劇本轉語音(ONNX 版)

與 `script_to_audio.py` 功能相同, 但以 `onnxruntime` 取代 PyTorch 做神經網路推論。需先執行過 `test_onnx.py` 或 `test_onnx_pc.py` 產生 `kokoro.onnx`。

```bash
.venv/bin/python script_to_audio_onnx.py example_story.txt
```

---

## PyTorch vs ONNX 效能比較

在 Intel Mac(macOS 12.7, x86_64, CPU only)上的實測結果 :

| | PyTorch | ONNX |
|---|---|---|
| 整體 RTF | 5.9x | 1.5x |
| 熱身後每句 RTF | 3-8x | 1.1-1.4x |
| 總耗時(7 句, 81s 音訊) | 483s | 123s |
| 加速倍率 | - | 約 4x |

在 Windows(CPU only, torch 2.12)上的實測結果 :

| | PyTorch | ONNX |
|---|---|---|
| RTF | 0.575x | 0.182x |

> 兩個平台都是 ONNX 版明顯較快, 但 Windows 上 torch 本身也因為版本較新而快上不少, PyTorch 路徑已接近即時合成。

---

## 範例劇本

`example_story.txt` : 三個角色(speaker1/2/3)的英文短篇故事, 標籤格式, 來自 TTS-Story 專案範例。

```
[speaker1]Once upon a time...[/speaker1]
[speaker2]One day...[/speaker2]
[speaker3]"I am the Guardian..."[/speaker3]
```
