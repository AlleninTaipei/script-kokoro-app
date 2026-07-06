# script-kokoro-app 學習筆記

## 文件產生資訊

本文件由 Claude Code Skills 的 learndoc 指令產生.
使用模型: Claude Sonnet 5.
產生日期: 2026 年 7 月 6 日.

---

## 這個專案到底在做什麼

想像你手上有一份多角色劇本, 你想把它變成一齣有聲廣播劇, 每個角色都有自己的聲音, 語氣自然, 不需要請配音員, 也不需要付費給雲端 TTS API。這個 repo 就是在做這件事。

它建立在 [hexgrad/kokoro](https://github.com/hexgrad/kokoro) 這個開源 TTS 引擎之上。Kokoro 本身只是一個 82M 參數的語音合成模型, 這個 repo 則是把它包裝成一套可以真正拿來用的工具, 從最原始的 命令列腳本, 到一個完整的網頁應用程式, 一路把 相容性測試, 效能優化, 多角色配音, LLM 自動選聲音 這幾件事串起來。

如果你把整個專案想成一間小型配音工作室, 這份筆記就是帶你走過這間工作室每個房間的導覽。

---

## 為什麼會有這個資料夾

從 git 紀錄可以看到, 這個 repo 是刻意從一個較舊的專案重新整理過來的單一 commit(`Initial commit: script-kokoro-app`)。也就是說, 現在你看到的每一支腳本, 都是已經在舊環境驗證過, 值得留下來的部分, 不是從零開始亂長出來的雛型。

`idea_0422.md` 這份規劃筆記透露了更長遠的野心: 用 Kokoro 產生的 MP3, 透過 GitHub Releases 加 GitHub Pages 架一個 Podcast RSS, 提交到 Spotify 與 Apple Podcasts。換句話說, 這些 CLI 腳本與 Web UI 不只是技術展示, 而是要撐起一整條 劇本 到 有聲節目 到 訂閱平台 的生產線的其中一段。理解這個大方向, 你就能明白為什麼程式碼裡處處在意 RTF(合成速度), WAV 與 MP3 輸出格式, 還有多角色配音品質, 因為這些都是決定一個 podcast 能不能穩定產出的關鍵細節。

---

## 檔案地圖

| 檔案 / 目錄 | 角色 |
|---|---|
| `test_mac.py` | Mac 環境健檢: Kokoro 能不能在這台機器上正常跑(PyTorch 路徑) |
| `test_pc.py` | 同上, 但是 Windows 版本, 播放音訊改用 `winsound` |
| `test_onnx.py` | Mac 版 ONNX 匯出與推論驗證, 順便產生 `kokoro.onnx` |
| `test_onnx_pc.py` | Windows 版 ONNX 匯出與推論驗證 |
| `script_to_audio.py` | 劇本轉語音, PyTorch 推論版, 支援多角色 |
| `script_to_audio_onnx.py` | 劇本轉語音, ONNX 推論版, 邏輯與上面完全對稱 |
| `web_app/app.py` | FastAPI 後端: 上傳劇本, LLM 自動配音, SSE 串流合成, 下載 |
| `web_app/static/index.html` | 瀏覽器前端介面 |
| `example_story.txt` | 標籤格式(`[speaker1]...[/speaker1]`)的英文範例劇本 |
| `sample_voices.fountain` / `sample_voices.txt` | Fountain 格式(好萊塢標準劇本格式)的範例劇本 |
| `idea_0422.md` | Podcast RSS 產品規劃筆記, 說明這個專案的長期目標 |
| `requirements.txt` / `requirements-mac-legacy.txt` | 依賴版本清單, 後者是給 Intel Mac 專用的版本釘選 |

一眼就能看出一個規律: 每個功能幾乎都有 兩份 甚至 三份 平行實作(Mac 版與 Windows 版, PyTorch 版與 ONNX 版, 舊解析格式與新解析格式)。這不是重複勞動, 而是刻意的 對照組設計, 之後會展開說明為什麼這樣做反而是聰明的做法。

---

## 整體架構: 一條配音工作室的生產線

把 `script_to_audio.py` 或 `script_to_audio_onnx.py` 想成一條電影配音的生產線:

1. 劇本督導(`parse_script`): 讀進劇本檔案, 拆解成一句一句的 `(角色, 台詞)`。
2. 選角導演(`assign_voices`): 幫每個角色分配一個 Kokoro 聲音 ID, 使用者也可以手動指定。
3. 配音員(Kokoro 引擎本身): 把每一句台詞真正合成出聲音。
4. 剪接師(`generate_script_audio`): 把每句話的音訊接起來, 中間補上短暫的靜音, 讓聽感自然。
5. 發行(`save_wav`): 把最後合成好的音訊寫成 `.wav` 檔案。

最值得注意的設計是, PyTorch 版和 ONNX 版的生產線幾乎一模一樣, 唯一不同的只有 配音員 這一站:

```
PyTorch 路徑:  KPipeline.__call__()  →  KModel.forward_with_tokens()  (torch CPU)
ONNX 路徑   :  KPipeline.g2p()       →  ort.InferenceSession.run()    (ONNX Runtime)
```

解析劇本, 分配聲音, 補靜音, 寫檔, 這些邏輯完全共用, 差別只在神經網路推論這一個環節。這是軟體設計裡一個很實用的原則: 把會變動的部分(推論引擎)和不會變動的部分(前後處理邏輯)切開, 之後想加第三種推論方式(例如未來的 GPU 加速版), 也不需要重寫整條生產線。

---

## 劇本解析器的演化: 從兩種格式到三種格式

CLI 腳本(`script_to_audio.py`)裡的 `parse_script` 只認得兩種格式:

格式 A, 對話格式:
```
Alice: Hello, how are you?
Bob: I'm doing great!
```

格式 B, 標籤格式, 跟 TTS Story 專案相容:
```
[alice]Hello, how are you?[/alice]
[bob]I'm doing great![/bob]
```

但等到 Web App(`web_app/app.py`)出現, 解析器多長出了第三種格式, Fountain 格式, 也就是好萊塢劇本業界的標準格式:

```
DR. CHEN
The waveform is unlike anything in our database.

ALICE
Could it be natural?
```

角色名稱是全大寫, 前面要有空行, 後面緊接著台詞。程式碼裡的 `_parse_fountain` 函式甚至處理了 `(V.O.)`(旁白), `(O.S.)`(畫外音), `(CONT'D)` 這類角色名稱的附加標記, 以及要跳過的舞台指示。

這個演化過程本身就是一堂課: 一開始你只需要處理自己寫的簡單格式, 但一旦要讓真實世界的使用者上傳劇本, 你就得面對別人早就用了幾十年的既有格式慣例。`parse_script` 的解法是 依序嘗試, 從最嚴謹, 最不會誤判的格式開始(Fountain 要求至少抓到兩個角色才採用), 一路降級到最寬鬆的冒號格式。這種 由嚴到寬 的嘗試順序, 比你直接猜哪種格式更穩定, 也更容易維護, 因為新增一種格式只需要在鏈的某個位置插入, 不用動到其他分支。

---

## ONNX 那條路: 為什麼值得多繞這一圈

### 問題出在哪裡

PyTorch 是研究用的框架, 靈活, 但不是為了極致的 CPU 推論速度而生。在資源有限的機器上(例如舊款 Intel Mac, 沒有 CUDA 也沒有 Apple Silicon 的 MPS 加速), 純 PyTorch 推論的 RTF(即時因子, Real Time Factor)可能落在 5 到 8 倍之間, 也就是合成 1 秒的語音要花 5 到 8 秒運算。對一句demo台詞還好, 對一整集 podcast 就是災難。

### 解法: 把模型匯出成 ONNX

ONNX Runtime 提供高度優化過的 CPU 核心運算, 繞過 PyTorch 本身的額外開銷。同一組模型權重, 換一個更精簡的執行環境跑, RTF 就能壓到 1 倍上下, 幾乎達到即時合成。這次驗證時實測(Windows, CPU only)的結果是:

| | PyTorch | ONNX |
|---|---|---|
| RTF | 0.18x | 0.20x |

有趣的是, 在這台 Windows 機器上兩者其實非常接近, 因為新版 torch 本身在 Windows 上優化得已經很好。但在舊款 Intel Mac 上(README 記錄的實測數字), 落差就很明顯: PyTorch 5.9x 對比 ONNX 1.5x, 差了將近 4 倍。這說明一個重要的觀念: 效能優化的效益高度依賴你的目標硬體, 同一套優化在不同機器上的報酬率可能完全不同, 所以每次拿到新環境都值得重新量測, 而不是直接沿用舊數字。

### 匯出過程踩過的坑

`test_onnx_pc.py` 裡有一行註解特別值得記住:

```python
torch.onnx.export(
    ...,
    dynamo=False,  # torch >= 2.6 預設改用新的 dynamo-based ONNX exporter,
                   # 對這個模型的動態控制流會匯出失敗
)
```

torch 2.6 之後預設的匯出器換成了以 `torch.compile` 為基礎的 dynamo exporter, 理論上更先進, 但這個模型內部有 資料相依的控制流(依照輸入長度決定要不要分段處理), dynamo exporter 目前處理不好這種情況, 匯出會直接失敗。解法是加上 `dynamo=False`, 強制退回舊版, 以 TorchScript 為基礎的匯出路徑。

這是個很實際的教訓: 框架的 新預設值 不代表對你的模型 一定更好, 遇到匯出失敗, 第一步該做的不是放棄 ONNX, 而是去查那個新預設值背後的假設是什麼, 你的模型是不是剛好踩到它的已知限制。

另外還有兩個容易被忽略的細節:

- `dynamic_axes` 告訴 ONNX, `input_ids` 這個輸入在不同呼叫之間長度可以不一樣(不同句子的音素數量本來就不同)。沒有這個設定, 匯出出來的模型就只認得跟範例輸入一模一樣長度的句子。
- `disable_complex=True` 是 Kokoro 提供的參數, 因為模型內部聲碼器(vocoder)用到複數張量做 STFT 運算, 而 ONNX 原生不支援 PyTorch 的複數運算, 這個參數會把複數運算換成等價的實數運算。

### `OnnxKokoro` 裡的依賴注入技巧

`script_to_audio_onnx.py` 裡最巧妙的一段設計是, 它建立了一個 `KModel` 物件, 但完全不用它做推論, 只用它來拿 詞彙表(vocab):

```python
# KModel 只用於取 vocab, 實際推論交給 ONNX Runtime
self.kmodel = KModel(repo_id="hexgrad/Kokoro-82M").eval()
self.vocab   = self.kmodel.vocab
```

接著把這個 `kmodel` 傳給 `KPipeline`, 讓 `KPipeline` 繼續負責 文字轉音素(G2P, 由 `misaki` 套件完成)跟 聲音包載入, 但真正的神經網路推論已經被抽換成 `ort.InferenceSession.run()`。這就是典型的 依賴注入(Dependency Injection): 抽換掉系統裡的一個元件(神經網路後端), 卻不用動到其他部分的程式碼。任何時候你發現自己在 建立一個物件, 只為了借用它的某個屬性或方法, 而刻意不去呼叫它最主要的功能, 那通常就是在做同樣的事情。

---

## 依賴地獄: 一個值得所有 Python 開發者記住的教訓

這個專案鎖定 Python 3.12, 而不是最新的 3.13, README 裡寫得很清楚: `kokoro` 依賴的 `misaki` 套件的 C 擴充目前還沒有 3.13 的 wheel。

在 Intel Mac(x86_64, macOS 12)上, 依賴鏈會像骨牌一樣一路牽連下去:

| 套件 | 限制 | 原因 |
|---|---|---|
| `torch` | 最高 2.2.2 | PyPI 在 2.4 之後不再發佈 x86_64 macOS 的 wheel |
| `numpy` | 需小於 2 | torch 2.2.2 是對著 NumPy 1.x 的 ABI 編譯的 |
| `transformers` | 需小於 5 | transformers 5.x 要求 torch 2.4 以上 |

這也是為什麼 repo 裡會有一個獨立的 `requirements-mac-legacy.txt`, 只在 Intel Mac 上額外疊加安裝。Windows 或其他有完整 torch wheel 支援的平台則完全不需要這個檔案, 直接裝 `requirements.txt` 裡的最新版即可, 這次在 Windows 環境驗證時也確實一次到位, 沒有遇到版本衝突。

這裡的心法是: 當你的目標平台是比較冷門或比較舊的硬體組合時, 越早去實際測試完整依賴鏈越好。`pip install torch` 在 Intel Mac 上不會報錯, 它只會默默幫你裝一個舊版本, 然後在你完全想不到的地方(例如匯出 ONNX 時的某個 API 不相容)才爆出問題。與其等到那時候再回頭排查, 不如一開始就把版本釘選寫清楚, 並且留一份文件說明 為什麼 要釘選這個版本, 這份 README 就做得很好, 每條限制都附了原因。

---

## Web App 深度剖析

如果說 CLI 腳本是這個專案的骨架, `web_app/app.py` 就是把整套能力包裝成一個真正能給別人用的產品。它是一個 FastAPI 應用, 搭配 SSE(Server Sent Events)做即時進度串流, 並且整合了雲端 LLM(可依 `.env` 的 `DEFAULT_PROVIDER` 切換 Anthropic / OpenAI / Google)做 自動配音導演。這一步完全是選用的, 使用者可以在網頁上取消勾選, 或者根本沒設定任何 API Key, 一樣能跑完整套多角色配音流程, 只是聲音改成依序輪詢分配, 再手動調整。

### 資料流: 從上傳劇本到下載成品

```
瀏覽器載入頁面
   → GET /config → 回傳目前是否已偵測到 API Key, 決定「使用 AI 自動配音」勾選框的預設狀態
瀏覽器上傳劇本(.txt)
   → POST /analyze(帶 use_ai 旗標)
        → use_ai=true 且呼叫 LLM 成功 → 依語氣挑聲音 → 回傳角色與聲音對照表, 附理由
        → use_ai=false, 或呼叫 LLM 失敗(含缺 API Key)→ 優雅降級成輪詢分配, 附上降級原因
   → 使用者可在角色配音表格手動調整
   → POST /synthesize → SSE 逐句推播進度(RTF, 角色, 目前第幾句)
   → 完成後回傳 session_id
   → GET /download/{session_id}?format=wav 或 mp3 → 下載成品
```

還有一個 `GET /preview/{voice_id}` 端點, 讓使用者在動手合成整份劇本之前, 可以先聽一小段示範文字, 判斷這個聲音適不適合, 而且第一次合成後的結果會被快取到 `web_app/tmp/preview_{voice_id}.wav`, 之後同一個聲音就不用重新算一次。

### `/analyze`: 讓 LLM 當配音導演(但完全是選用的)

這是整個 Web App 最有意思的一段。它把每個角色的台詞範例, 加上完整的聲音目錄(語言, 性別, 音色描述), 一起丟給雲端 LLM, 請它扮演 TTS 配音導演, 依照語言, 性別, 年齡, 語氣, 幫每個角色選出最合適的聲音, 並且要求回傳結構化的 JSON, 附上 一句話理由。

呼叫哪一家 LLM 由 `call_llm()` 依 `.env` 裡的 `DEFAULT_PROVIDER`(anthropic / openai / google)分派, 三個分支各自用對應的 SDK 建立 client, 讀取各自的 `XXX_API_KEY`。

這裡有幾個值得學的細節:

第一, prompt 裡明確要求 回傳純 JSON, 不含 markdown, 但程式碼並沒有天真地相信 LLM 一定會乖乖照做, 而是先做一次清理:

```python
raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw).strip()
```

LLM 常常會忍不住把 JSON 包在 ```json 這種 markdown 區塊裡, 即使 prompt 已經明講不要, 這行 regex 就是在防這個已知的行為模式。

第二, 也是後來補上的一課: 一開始這段邏輯只用 `except json.JSONDecodeError` 接住 LLM 回傳格式跑掉的情況, 但 `call_llm()` 內部是直接用 `os.environ["ANTHROPIC_API_KEY"]` 這種寫法讀金鑰, 沒設定就會丟 `KeyError`, 而這個例外完全沒被接住, 一路往上炸成 500。這正好是這個專案一開始就想避免, 又不小心留下的坑, 因為使用者可能出於各種理由(例如公司 IT 對雲端服務金鑰有稽核考量)不想設定任何 API Key, 這種情況下多角色配音應該退化成 手動指定, 而不是直接壞掉。

解法是把攔截範圍從單一 `json.JSONDecodeError` 放大成整段 `try/except Exception`, 涵蓋 呼叫 LLM 本身失敗, 缺 Key 的 `KeyError`, 額度不足, 逾時, 回傳格式錯誤 等各種情況, 全部退回跟 CLI 版一樣的輪詢式自動分配:

```python
try:
    raw = call_llm(prompt).strip()
    raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw).strip()
    result = json.loads(raw)
    voice_map = {c: v for c, v in result.get("mapping", {}).items() if v in VOICE_CATALOG}
    ...
    ai_used = True
except Exception as e:
    print(f"AI 自動配音失敗（將改用手動輪詢分配）：{e}")
    ai_error = str(e)
```

同一段改動也順手加了一個 `use_ai` 旗標, 讓前端可以直接 主動 跳過 LLM, 不用等它失敗才降級, 並且新增了一個很小的 `GET /config` 端點, 只回傳「目前 provider 是否已偵測到 API Key」, 讓網頁一載入就知道要不要把「使用 AI 自動配音」這個勾選框預設打開。回應多了 `ai_used` / `ai_error` 兩個欄位, 前端可以據此在角色配音表格上方顯示一則提示, 讓使用者知道 這次到底有沒有真的用到 AI, 而不是誤以為勾了選項就一定生效。

這是一個很好的 優雅降級(graceful degradation)範例, 也是這個專案第二次用同一個心法解決問題(第一次是 ffmpeg 缺失導致 MP3 轉檔失敗, 見下文): 呼叫外部服務永遠有失敗的可能, 好的系統設計不是假設它一定成功, 而是先想好 如果它失敗了(或使用者根本不想用它), 使用者體驗會退化到哪裡, 而不是直接壞掉或被擋在門外。

### `/synthesize`: SSE 串流背後的執行緒協作

`/synthesize` 的實作用了一個常見但值得拆解的模式: 真正跑語音合成的 `run_tts()` 函式是同步, 阻塞式的程式碼(合成語音本來就是吃 CPU 的重工作), 但 FastAPI 的路由函式是 `async def`。要在 async 的世界裡跑一段會佔用 CPU 好幾秒到好幾分鐘的同步工作, 又不讓整個伺服器被卡住, 標準做法是丟進一個獨立執行緒:

```python
loop.run_in_executor(concurrent.futures.ThreadPoolExecutor(max_workers=1), run_tts)
```

`run_tts()` 每完成一句台詞, 就透過 `loop.call_soon_threadsafe(queue.put_nowait, event)` 把一則進度事件塞進 `asyncio.Queue`, 而真正回傳給瀏覽器的 `event_stream()` 則是一個 async generator, 不斷從 queue 裡撈事件, 轉成 SSE 格式的文字丟出去。這一整套設計讓 CPU 密集的推論工作, 跟 I/O 密集的網路串流, 各自用最適合自己的方式執行, 中間只靠一個執行緒安全的 queue 溝通, 彼此不會互相卡住。

### 我們在驗證時抓到的真實 bug

這段特別值得記錄, 因為它不是憑空想像的案例, 而是這次實際跑測試時發生的事。

在沒有安裝 `ffmpeg` 的機器上呼叫 `/synthesize`, 語音合成本身完全正常, 每一句的 SSE 進度事件都正確送出, WAV 檔案也確實成功寫進 `web_app/tmp/`。但瀏覽器那端會永遠停在轉圈, 收不到代表完成的 `done` 事件, 一路卡到 600 秒逾時。

根因藏在原本的程式碼裡:

```python
(TEMP_DIR / f"{session_id}.wav").write_bytes(wav_data)
(TEMP_DIR / f"{session_id}.mp3").write_bytes(wav_bytes_to_mp3(wav_data))
```

`wav_bytes_to_mp3()` 內部會呼叫 `subprocess.run(["ffmpeg", ...])`, 如果系統找不到 `ffmpeg` 執行檔, 這裡會拋出 `FileNotFoundError`。但這整段程式碼是跑在背景執行緒裡(前面提到的 `run_in_executor`), 而呼叫端從來沒有去 `await` 那個 future 的結果, 所以例外訊息只會安靜地印在伺服器的 log 裡(`Future exception was never retrieved`), 不會有任何東西通知前端。更關鍵的是, 因為例外中斷了 `run_tts()` 的執行, 後面那兩行原本要送出的 `done` 事件跟 sentinel(`None`), 根本沒有機會執行, SSE 連線就這樣被晾在那裡, 直到逾時。

這其實是把兩個各自看起來很小的問題疊在一起才會爆炸: 一個是 選用的功能(MP3 轉檔)失敗, 卻拖累了 核心功能(WAV 合成)的完成通知, 另一個是 背景執行緒的例外沒有人接手處理。修法很直接, 把選用的那一段包進 try/except, 失敗就印警告訊息, 繼續往下走, 讓 `done` 事件一定會被送出:

```python
try:
    (TEMP_DIR / f"{session_id}.mp3").write_bytes(wav_bytes_to_mp3(wav_data))
except Exception as e:
    print(f"MP3 轉檔失敗（將只提供 WAV 下載）：{e}")
```

修好之後重新測試, `done` 事件正常送達, WAV 下載回傳 200, MP3 下載則正確回傳 404(因為檔案真的沒有產生), 而不是讓整個請求卡死。

這裡有一個很值得內化的原則: 當一個流程裡有 必要步驟 跟 錦上添花的步驟, 千萬不要讓後者的失敗擋住前者的結果送達使用者手上。更廣一層說, 任何丟進背景執行緒或 `run_in_executor` 的工作, 都要認真想過 如果它半路丟例外, 誰會知道, 使用者會看到什麼, 而不是預設它一定會順利跑完。這種 沉默失敗(silent failure)特別危險, 因為在開發機上你可能剛好裝了 ffmpeg, 完全不會發現這個問題, 一直到某個使用者的機器上沒裝, 才會在你完全沒設防的地方炸開。

---

## RTF: 怎麼讀懂語音合成的速度數字

RTF, 全名 Real Time Factor, 即時因子, 是 TTS 領域最常用的速度指標:

```
RTF = 推論耗時 / 音訊時長
```

- RTF 小於 1.0, 代表比即時還快, 合成 10 秒的音訊花不到 10 秒, 這種速度才適合拿來做線上互動應用。
- RTF 等於 1.0, 代表剛好即時。
- RTF 大於 1.0, 代表比即時慢, 適合離線批次處理, 不適合即時對話場景。

程式碼裡幾乎每一支腳本都會逐句印出 RTF, 這不是為了炫技, 而是讓你在合成一整份長劇本時, 能一眼看出 是不是某一句特別慢, 通常代表那句特別長, 或是冷啟動剛好發生在那一句(第一次呼叫某個語言的 pipeline 需要先載入模型)。這次在 Web App 測試裡就實際觀察到這個現象: 同一個 backend, 第一句的 RTF 明顯比第二句高, 原因就是第一句剛好撞上 ONNX session 或 pipeline 的冷啟動。

---

## 工程心法整理

### 先量測, 再優化

ONNX 帶來的加速不是靠猜測得來的, 而是先老老實實把 PyTorch 版本的 RTF 量出來, 才知道 到底值不值得多花力氣做一次模型匯出。沒有那個基準數字, 你根本無從判斷 ONNX 到底有沒有用。

### 熱路徑要精簡

`OnnxKokoro._run_segment` 或 `web_app/app.py` 裡的 `_segment` 函式, 核心那幾行 `session.run(...)` 完全沒有多餘的錯誤處理, 沒有 log, 沒有重試邏輯。這是刻意的: 迴圈裡每多一行 Python 運算都會拖慢整體延遲, 錯誤處理應該放在外層的邊界, 而不是塞進被呼叫成千上萬次的內層迴圈。

### 該偷懶的地方就偷懶: 延遲初始化

不管是 `KPipeline` 還是 ONNX session, 全部都是 用到才建立:

```python
if lang_code not in self._pipelines:
    self._pipelines[lang_code] = KPipeline(lang_code=lang_code, model=self.kmodel)
```

一份只有英文角色的劇本, 就永遠不會去載入中文的 pipeline, 省下實際的模型下載時間跟記憶體。

### WAV 格式其實很單純

整個專案反覆出現同一段寫檔邏輯:

```python
wf.setnchannels(1)
wf.setsampwidth(2)
wf.setframerate(24000)
wf.writeframes((audio * 32767).astype(np.int16).tobytes())
```

模型輸出的是 `float32`, 範圍在負 1 到正 1 之間, 換成 16 位元 PCM 只需要乘上 `32767`(有號 16 位元整數的最大值)再轉型別, Python 內建的 `wave` 模組會處理剩下所有細節。這是一個 看起來很底層, 其實一點都不難 的例子, 很多人會誤以為音訊格式處理很麻煩而不敢碰。

### 測試腳本本身就是文件

`test_mac.py` 跟 `test_pc.py` 不只是測試, 更是 可以直接執行的說明書。任何人拿到這個 repo, 跑一次 `test_pc.py`, 30 秒內就知道自己的環境到底行不行, 每一步都印出清楚的 `[OK]` 或 `[FAIL]`, 失敗時還附上修復提示(例如 `pip install kokoro`)。這種寫法比一大段 README 文字更有效, 因為它會 主動告訴你哪裡壞了, 而不是要你自己去猜。

### 長句子的處理: 承認限制, 但不能讓程式當掉

Kokoro 模型每段輸入最多接受 510 個音素。英文靠 `KPipeline.en_tokenize()` 在自然的標點或子句邊界自動切段, 但其他語言目前是硬截斷:

```python
if len(ps) > 510:
    ps = ps[:510]
```

這是一個 承認目前做不到完美, 但先求不當機 的權衡取捨。硬截斷可能會在句子中間被切掉, 聽起來不完整, 但至少不會讓整支程式崩潰。這種 已知的技術債, 值得直接寫在程式碼裡當作註解, 讓下一個接手的人知道 這裡還有進步空間, 而不是假裝問題不存在。

### 選用功能不該拖垮核心功能

這是這次驗證過程裡新學到, 而且是用真實 bug 換來的心法: MP3 轉檔是加分項, WAV 合成才是核心功能。任何時候你在設計一個 主要結果加上附加加工 的流程, 都要問自己, 如果附加加工失敗了, 主要結果還能不能送達使用者手上。

---

## 未來可能踩到的坑

- 如果之後要接上 GPU(CUDA 或 Apple Silicon 的 MPS), RTF 有機會壓到 0.3 倍以下, 但要注意 ONNX Runtime 的 GPU 版本(`onnxruntime-gpu`)跟目前用的 CPU 版本是不同的套件, 需要另外處理安裝與相容性。
- Fountain 格式解析器目前是用 全大寫 加上 前後空行 當作角色名稱的判斷依據, 如果劇本裡出現全大寫的舞台指示或章節標題, 有可能被誤判成角色, 之後若要處理更複雜的真實劇本, 這段判斷邏輯值得加強。
- `web_app/app.py` 的 `call_llm()` 目前用一串 if / elif 分派 anthropic / openai / google 三個供應商, 每多加一家就多一個分支跟一段重複的「讀 Key, 建 client, 呼叫」邏輯, 之後若要再加第四家, 值得改成 dict-of-callables 的 registry 模式, 而不是繼續往同一個函式塞 if。
- 任何丟進 `run_in_executor` 或背景執行緒的邏輯, 記得對照這次 ffmpeg 的教訓, 全程包好 try/except, 並且確保無論成功或失敗, 都要有辦法讓呼叫端知道結果, 不要讓例外靜靜消失。

---

## 如何執行整個專案

### CLI 腳本(Windows)

```powershell
# 環境設定
uv venv --python 3.12 .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 環境健檢
.venv/Scripts/python.exe test_pc.py

# 產生 ONNX 模型並驗證
.venv/Scripts/python.exe test_onnx_pc.py

# 劇本轉語音, PyTorch 版
.venv/Scripts/python.exe script_to_audio.py example_story.txt

# 劇本轉語音, ONNX 版, 約快 4 倍
.venv/Scripts/python.exe script_to_audio_onnx.py example_story.txt
```

### Web UI

```powershell
# (選用)建立 .env, 依 .env.example 設定 DEFAULT_PROVIDER 與對應的 API Key
# 啟動 server
.venv/Scripts/python.exe -m uvicorn web_app.app:app --port 8000

# 瀏覽器開啟 http://localhost:8000
```

沒有設定 `.env` 也能正常使用多角色配音, 網頁會自動偵測沒有 API Key, 把「使用 AI 自動配音」勾選框預設關閉, 改用手動指定聲音的方式繼續走完整套流程。

MP3 下載功能需要系統另外安裝 `ffmpeg` 並加入 PATH, 沒有安裝的話, 系統現在會優雅降級成只提供 WAV 下載, 而不會讓整個合成請求卡住。
