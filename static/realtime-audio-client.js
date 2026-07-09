/* ── 环形缓冲区 ── */
class AudioRingBuffer {
  constructor(maxDurationMs = 120000) {
    this.maxDurationMs = maxDurationMs;
    this.chunks = [];
    this.totalDurationMs = 0;
    this.sampleRate = null;
  }
  push(pcmBuffer, sampleRate, chunkDurationMs) {
    this.sampleRate = sampleRate;
    this.chunks.push({ pcm: pcmBuffer.slice(0), durationMs: chunkDurationMs });
    this.totalDurationMs += chunkDurationMs;
    while (this.totalDurationMs > this.maxDurationMs && this.chunks.length > 1) {
      this.totalDurationMs -= this.chunks[0].durationMs;
      this.chunks.shift();
    }
  }
  getSampleRate() { return this.sampleRate; }
  getChunks() { return this.chunks.map((c) => c.pcm); }
  getTotalDurationMs() { return this.totalDurationMs; }
  clear() { this.chunks = []; this.totalDurationMs = 0; this.sampleRate = null; }
}


export class RealtimeAudioClient extends EventTarget {
  constructor(options = {}) {
    super();
    this.options = {
      healthUrl: "/health",
      recorderWorkletUrl: "/static/recorder-worklet.js",
      getWebSocketUrl: defaultWebSocketUrl,
      getModel: () => "",
      getPrompt: () => "",
      getPrefillMs: () => 600,
      getRouteContext: () => "",
      getConversationMode: () => "manual",
      getSessionId: () => "",
      getOutputAudio: () => false,
      getStreamSpeechAudio: () => false,
      getEasyTurnEnabled: () => false,
      getVadConfig: () => null,
      getBargeInVadConfig: () => null,
      shouldAutoStartOnVad: () => false,
      getVadMonitorWebSocketUrl: defaultVadMonitorWebSocketUrl,
      shouldAutoStopOnVad: () => true,
      useWebAudioPlayback: false,
      enableBargeIn: false,
      bargeInAfterFinalResultOnly: false,
      bargeInChunkMs: 120,
      bargeInCooldownMs: 800,
      bargeInPreBufferMs: 3000,
      autoVad: false,
      ...options,
    };
    this.state = {
      ws: null,
      startedAt: 0,
      chunks: 0,
      prefillOk: 0,
      audioQueue: [],
      audioPlaying: false,
      currentAudio: null,
      audioTextByIndex: new Map(),
      playedAssistantText: "",
      finalReceived: false,
      finalText: "",
      finalHistoryText: "",
      finalUserText: "",
      resumeAfterTurnAck: false,
      interruptedAssistantText: "",
      interrupted: false,
      streamedAudioCount: 0,
      streamingTextStarted: false,
      mode: "idle",
      timer: null,
      flushResolve: null,
      playbackContext: null,
      playbackGeneration: 0,
      bargeInTriggered: false,
      preBufferChunks: null,
      preBufferSampleRate: null,
      // persistent mic — always on
      _persistentMic: null,
      _persistentMicSampleRate: null,
      _persistentVadWs: null,
      _ringBuffer: null,
      _sendingAudio: false,
      vad: this.createVadState(),
    };
    this.setMode("idle");
  }

  createVadState() {
    return {
      enabled: Boolean(this.options.autoVad),
      speechStarted: false,
      autoStopping: false,
      speechMs: 0,
      silenceMs: 0,
      totalSpeechMs: 0,
      latestRms: 0,
    };
  }

  on(type, listener) { this.addEventListener(type, (event) => listener(event.detail)); }
  emit(type, detail = {}) { this.dispatchEvent(new CustomEvent(type, { detail })); }
  log(message, level = "") { this.emit("log", { message, level }); }
  setStatus(text, ok = false) { this.emit("status", { text, ok }); }
  setVoiceState(text) { this.emit("voiceState", { text }); }
  setMode(mode) { this.state.mode = mode; this.emit("mode", { mode }); }
  emitMetrics() {
    const durationSeconds = this.state.startedAt ? (performance.now() - this.state.startedAt) / 1000 : 0;
    this.emit("metrics", { chunks: this.state.chunks, prefillOk: this.state.prefillOk, durationSeconds });
  }

  async loadServerConfig() {
    const response = await fetch(this.options.healthUrl);
    if (!response.ok) throw new Error(`health check failed: ${response.status}`);
    const config = await response.json();
    this.emit("config", { config });
    this.log(`config model=${config.model} provider=${config.provider} prefill=${config.prefill_mode}`);
    return config;
  }

  async handleAction() {
    if (this.state.mode === "idle") { await this.startRecording(); }
    else if (this.state.mode === "recording") { await this.stopRecording(); }
    else if (this.state.mode === "finalizing" || this.state.mode === "speaking") { await this.interruptAndRecord(); }
  }

  // ═══════════════════════════════════════════════════════════
  //  Persistent mic — always on, feeds ring buffer + VAD
  // ═══════════════════════════════════════════════════════════

  async _ensurePersistentMic() {
    if (!this.state._persistentMic) {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      const audioContext = new AudioContext({ latencyHint: "interactive" });
      await audioContext.audioWorklet.addModule(this.options.recorderWorkletUrl);
      const source = audioContext.createMediaStreamSource(stream);
      const chunkMs = Number(this.options.getPrefillMs() || 600);
      const node = new AudioWorkletNode(audioContext, "pcm-chunker", {
        processorOptions: { chunkMs },
      });
      const sink = audioContext.createGain();
      sink.gain.value = 0;
      source.connect(node);
      node.connect(sink);
      sink.connect(audioContext.destination);

      const ringBuf = new AudioRingBuffer(Number(this.options.bargeInPreBufferMs || 120000));
      this.state._ringBuffer = ringBuf;

      node.port.onmessage = (event) => {
        if (!event.data) return;
        if (event.data.type === "flushed") { this.resolveFlush(); return; }
        if (event.data.type !== "chunk") return;
        ringBuf.push(event.data.pcm, audioContext.sampleRate, chunkMs);
        if (this.state._persistentVadWs && this.state._persistentVadWs.readyState === WebSocket.OPEN) {
          this.state._persistentVadWs.send(event.data.pcm);
        }
        if (this.state._sendingAudio && this.state.ws && this.state.ws.readyState === WebSocket.OPEN) {
          this.state.ws.send(event.data.pcm);
          this.state.chunks += 1;
          this.emitMetrics();
        }
      };

      this.state._persistentMic = { stream, audioContext, source, node, sink };
      this.state._persistentMicSampleRate = audioContext.sampleRate;
      this.log(`persistent mic started, sampleRate=${audioContext.sampleRate}`);
    }

    if (!this.options.shouldAutoStartOnVad() || this.state._persistentVadWs) return;

    const audioContext = this.state._persistentMic.audioContext;
    const vadConfig = this.options.getVadConfig() || {
      engine: "silero", threshold: 0.8, minSpeechMs: 128, minSilenceMs: 800, maxSpeechMs: 30000, speechPadMs: 30,
    };
    const vadWs = new WebSocket(this.options.getVadMonitorWebSocketUrl());
    this.state._persistentVadWs = vadWs;
    vadWs.binaryType = "arraybuffer";
    vadWs.onmessage = (event) => this._onPersistentVadEvent(event, vadWs);
    vadWs.onerror = () => this.log("persistent VAD WS error", "error");
    await new Promise((resolve, reject) => {
      vadWs.onopen = resolve;
      vadWs.onclose = () => {
        if (this.state._persistentVadWs === vadWs) this.state._persistentVadWs = null;
        reject(new Error("persistent VAD WS closed"));
      };
    });
    vadWs.send(JSON.stringify({ type: "start", sampleRate: audioContext.sampleRate, vad: vadConfig }));
    this.log("persistent VAD started");
  }

  _stopPersistentMic() {
    if (this.state._persistentVadWs) {
      try { this.state._persistentVadWs.close(); } catch (_) {}
      this.state._persistentVadWs = null;
    }
    const m = this.state._persistentMic;
    if (m) {
      m.node.disconnect(); m.source.disconnect(); m.sink.disconnect();
      m.stream.getTracks().forEach((t) => t.stop());
      m.audioContext.close().catch(() => {});
      this.state._persistentMic = null;
      this.state._persistentMicSampleRate = null;
    }
    if (this.state._ringBuffer) { this.state._ringBuffer.clear(); this.state._ringBuffer = null; }
  }

  _drainRingBuffer() {
    const buf = this.state._ringBuffer;
    if (!buf || !buf.getChunks().length) return;
    this.state.preBufferChunks = buf.getChunks();
    this.state.preBufferSampleRate = buf.getSampleRate();
    this.log(`drain buffer: ${buf.getChunks().length} chunks, ~${Math.round(buf.getTotalDurationMs())}ms`);
    buf.clear();
  }

  clearPassiveBufferAndResetVad() {
    if (this.state._ringBuffer) this.state._ringBuffer.clear();
    if (this.state._persistentVadWs && this.state._persistentVadWs.readyState === WebSocket.OPEN) {
      const vadConfig = this.options.getVadConfig() || {
        engine: "silero", threshold: 0.8, minSpeechMs: 128, minSilenceMs: 800, maxSpeechMs: 30000, speechPadMs: 30,
      };
      this.state._persistentVadWs.send(JSON.stringify({
        type: "reset_vad",
        sampleRate: this.state._persistentMicSampleRate || 48000,
        vad: vadConfig,
      }));
    }
  }

  // ═══════════════  Persistent VAD events  ═══════════════

  _onPersistentVadEvent(event, socket) {
    if (this.state._persistentVadWs !== socket) return;
    const data = JSON.parse(event.data);
    switch (data.type) {
      case "ready":
      case "vad_ready":
      case "vad_monitor_started":
        this.log(`persistent VAD ${data.type}`);
        break;
      case "vad_speech_start":
        if (this.options.shouldAutoStartOnVad() && (this.state.audioPlaying || this.state.mode === "finalizing" || this.state.mode === "speaking")) {
          const cooldown = Number(this.options.bargeInCooldownMs || 800);
          if (this.state.audioPlaying && performance.now() - (this.state.startedAt || 0) < cooldown) {
            this.log("barge-in ignored during cooldown"); return;
          }
          this.emit("bargeIn", { probability: data.probability, timeMs: data.time_ms });
          this._doBargeIn().catch((err) => {
            this.log(`barge-in failed: ${err.message || err}`, "error");
            this.setMode("idle");
          });
        } else if (this.options.shouldAutoStartOnVad() && this.state.mode === "idle") {
          this._doIdleStart().catch((err) => {
            this.log(`idle start failed: ${err.message || err}`, "error");
            this.setMode("idle");
          });
        }
        break;
      case "vad_speech_end":
        this.setVoiceState("检测到说完，正在推理");
        this.emit("vad", { event: data.event || "speech_end", speaking: false, probability: data.probability });
        if (this.options.shouldAutoStopOnVad() && this.state.mode === "recording") {
          this.stopRecording().catch((err) => {
            this.log(`vad stop error: ${err.message || err}`, "error");
            this.setMode("idle");
          });
        }
        break;
      case "vad_error":
      case "error":
        this.log(`persistent VAD error: ${data.message || data.type}`, "error");
        break;
    }
  }

  async _doBargeIn() {
    if (!this.options.enableBargeIn || this.state.bargeInTriggered) return;
    this.state.bargeInTriggered = true;
    this.state.interrupted = true;
    if (this.state.playedAssistantText.trim()) {
      this.state.interruptedAssistantText = this.state.playedAssistantText.trim();
    }
    this._drainRingBuffer();
    this.stopPlayback();
    this.closeWs();
    this.setMode("busy");
    this.setStatus("检测到插话，开始录音", true);
    this.setVoiceState("正在听你说");
    await this.startRecording();
  }

  async _doIdleStart() {
    this._drainRingBuffer();
    this.setMode("busy");
    this.setStatus("检测到说话", true);
    this.setVoiceState("正在听你说");
    await this.startRecording();
  }

  // ═══════════════  Recording (WS-based, uses persistent mic)  ═══════════════

  async startRecording() {
    this.setMode("busy");
    this.emit("result", { text: "录音中。", replace: true });
    this.setVoiceState("倾听中");
    this.emit("clearLog");

    // Ensure persistent mic is running (first call initializes it)
    await this._ensurePersistentMic();

    // Drain ring buffer (captured during AI playback / gap)
    this._drainRingBuffer();

    this.state.chunks = 0;
    this.state.prefillOk = 0;
    this.state.streamedAudioCount = 0;
    this.state.streamingTextStarted = false;
    this.state.audioTextByIndex = new Map();
    this.state.playedAssistantText = "";
    this.state.finalReceived = false;
    this.state.finalText = "";
    this.state.finalHistoryText = "";
    this.state.finalUserText = "";
    this.state.resumeAfterTurnAck = false;
    this.state.interrupted = false;
    this.state.bargeInTriggered = false;
    this.state.vad = this.createVadState();
    this.emitMetrics();

    const socket = new WebSocket(this.options.getWebSocketUrl());
    this.state.ws = socket;
    socket.binaryType = "arraybuffer";
    socket.onmessage = (event) => this.onServerEvent(event, socket);
    socket.onerror = () => this.log("WebSocket 错误", "error");
    await new Promise((resolve, reject) => {
      socket.onopen = resolve;
      socket.onclose = () => reject(new Error("WebSocket closed before start"));
    });

    const sr = this.state._persistentMicSampleRate || 48000;
    const chunkMs = Number(this.options.getPrefillMs() || 600);
    const interruptedText = this.state.interruptedAssistantText;
    this.state.interruptedAssistantText = "";

    socket.send(JSON.stringify({
      type: "start",
      sampleRate: sr,
      prefillMs: chunkMs,
      model: this.options.getModel().trim(),
      prompt: this.options.getPrompt().trim(),
      routeContext: this.options.getRouteContext(),
      conversationMode: this.options.getConversationMode(),
      session_id: this.options.getSessionId(),
      outputAudio: Boolean(this.options.getOutputAudio()),
      streamSpeechAudio: Boolean(this.options.getStreamSpeechAudio()),
      easyTurnEnabled: Boolean(this.options.getEasyTurnEnabled()),
      vad: this.options.getVadConfig(),
      interrupted_assistant_text: interruptedText || "",
    }));

    // Send pre-buffer chunks first
    const preChunks = this.state.preBufferChunks;
    const preRate = this.state.preBufferSampleRate;
    this.state.preBufferChunks = null;
    this.state.preBufferSampleRate = null;
    if (preChunks && preChunks.length) {
      for (let i = 0; i < preChunks.length; i += 1) { socket.send(preChunks[i]); }
      this.state.chunks += preChunks.length;
      this.log(`sent ${preChunks.length} pre-buffer chunks (${preRate || "?"}Hz)`);
    }

    this.state._sendingAudio = true;
    this.state.startedAt = performance.now();
    this.state.timer = setInterval(() => this.emitMetrics(), 100);
    this.setStatus("录音中", true);
    this.setMode("recording");
    this.log(`recording started, sampleRate=${sr}`);
    if (this.options.autoVad) {
      this.setVoiceState("等待说话");
      this.emit("vad", { event: "waiting", speaking: false, rms: 0 });
    }
  }

  async stopRecording() {
    if (this.state.mode !== "recording") return;
    this.setMode("finalizing");
    this.setStatus("停止录音，等待最终推理");
    this.setVoiceState("模型输出中");
    await this.flushInput();
    this.state._sendingAudio = false;
    if (this.state.ws && this.state.ws.readyState === WebSocket.OPEN) {
      this.state.ws.send(JSON.stringify({ type: "stop" }));
    }
    if (this.state.timer) clearInterval(this.state.timer);
    this.state.timer = null;
  }

  async interruptAndRecord() {
    if (this.state.mode !== "finalizing" && this.state.mode !== "speaking") return;
    this.state.interrupted = true;
    if (this.state.playedAssistantText.trim()) {
      this.state.interruptedAssistantText = this.state.playedAssistantText.trim();
    }
    this._drainRingBuffer();
    this.stopPlayback();
    this.closeWs();
    this.setMode("busy");
    this.setStatus("已打断，准备录音", true);
    this.setVoiceState("倾听中");
    await this.startRecording();
  }

  async bargeInAndRecord() { return this._doBargeIn(); }

  async startPassiveCapture() {
    await this._ensurePersistentMic();
  }

  stopBargeInMonitor() {
    this._stopPersistentMic();
  }

  // ═══════════════  Audio playback  ═══════════════

  async unlockAudioOutput() {
    if (!this.options.useWebAudioPlayback) return;
    const context = this.ensurePlaybackContext();
    if (context.state === "suspended") await context.resume();
    const source = context.createBufferSource();
    source.buffer = context.createBuffer(1, 1, context.sampleRate);
    source.connect(context.destination);
    source.start();
  }

  ensurePlaybackContext() {
    if (!this.state.playbackContext) {
      const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
      if (!AudioContextCtor) throw new Error("WebAudio is not supported by this browser");
      this.state.playbackContext = new AudioContextCtor();
    }
    return this.state.playbackContext;
  }

  stopPlayback() {
    this.state.playbackGeneration += 1;
    if (this.state.currentAudio) {
      this.state.currentAudio.onended = null;
      this.state.currentAudio.onerror = null;
      if (typeof this.state.currentAudio.pause === "function") {
        this.state.currentAudio.pause();
      } else if (typeof this.state.currentAudio.stop === "function") {
        try { this.state.currentAudio.stop(); } catch (_err) { /* already stopped */ }
      }
      this.state.currentAudio = null;
    }
    this.state.audioQueue = [];
    this.state.audioPlaying = false;
  }

  enqueueAudio(audioDataUrl, audioIndex, historyText = "") {
    if (!audioDataUrl) return;
    const index = Number(audioIndex || this.state.streamedAudioCount || 1);
    this.state.audioTextByIndex.set(index, historyText || this.getDisplayedText());
    this.state.audioQueue.push({ url: audioDataUrl, index });
    if (!this.state.audioPlaying) this.playNextAudio();
  }

  async playStandaloneAudio(audioDataUrl, historyText = "") {
    if (!audioDataUrl) return;
    if (this.options.shouldAutoStartOnVad()) await this.startPassiveCapture();
    this.state.finalReceived = true;
    this.state.interrupted = false;
    this.state.playedAssistantText = historyText || this.getDisplayedText();
    this.enqueueAudio(audioDataUrl, this.state.streamedAudioCount + 1, historyText || this.getDisplayedText());
    if (this.hasPendingPlayback()) this.setMode("speaking");
  }

  playNextAudio() {
    const next = this.state.audioQueue.shift();
    if (!next) {
      // Keep persistent mic+VAD alive for next turn detection
      this.state.audioPlaying = false;
      this.state.currentAudio = null;
      if (this.state.resumeAfterTurnAck) {
        this.resumeRecordingAfterTurnAck();
        return;
      }
      if (this.state.finalReceived && !this.state.interrupted) {
        if (this.options.shouldAutoStartOnVad()) this.clearPassiveBufferAndResetVad();
        this.setMode("idle");
      }
      this.setVoiceState("播报完");
      return;
    }
    this.state.audioPlaying = true;
    this.setVoiceState("正在播报中");
    if (this.options.useWebAudioPlayback) {
      const generation = this.state.playbackGeneration;
      this.playNextAudioWithWebAudio(next, generation).catch((err) => {
        this.log(`web audio play error: ${err.message || err}`, "error");
        if (generation === this.state.playbackGeneration) this.playNextAudioWithElement(next);
      });
      return;
    }
    this.playNextAudioWithElement(next);
  }

  async playNextAudioWithWebAudio(next, generation) {
    const context = this.ensurePlaybackContext();
    if (context.state === "suspended") await context.resume();
    const arrayBuffer = await this.audioDataUrlToArrayBuffer(next.url);
    const audioBuffer = await context.decodeAudioData(arrayBuffer);
    if (generation !== this.state.playbackGeneration || !this.state.audioPlaying) return;
    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);
    this.state.currentAudio = source;
    source.onended = () => {
      if (this.state.currentAudio === source) this.state.currentAudio = null;
      this.state.playedAssistantText =
        this.state.audioTextByIndex.get(next.index) || this.state.playedAssistantText || this.getDisplayedText();
      this.playNextAudio();
    };
    source.start();
  }

  playNextAudioWithElement(next) {
    const player = new Audio(next.url);
    this.state.currentAudio = player;
    player.onended = () => {
      if (this.state.currentAudio === player) this.state.currentAudio = null;
      this.state.playedAssistantText =
        this.state.audioTextByIndex.get(next.index) || this.state.playedAssistantText || this.getDisplayedText();
      this.playNextAudio();
    };
    player.onerror = () => {
      this.log("audio chunk decode/play error, skip to next", "error");
      if (this.state.currentAudio === player) this.state.currentAudio = null;
      this.playNextAudio();
    };
    player.play().catch((err) => {
      this.log(`audio play blocked: ${err.message || err}`, "error");
      if (this.state.currentAudio === player) this.state.currentAudio = null;
      setTimeout(() => this.playNextAudio(), 0);
    });
  }

  getDisplayedText() {
    return this.options.getDisplayedText ? this.options.getDisplayedText() : "";
  }

  async audioDataUrlToArrayBuffer(dataUrl) {
    const commaIndex = dataUrl.indexOf(",");
    if (dataUrl.startsWith("data:") && commaIndex >= 0) {
      const header = dataUrl.slice(0, commaIndex);
      const body = dataUrl.slice(commaIndex + 1);
      if (header.includes(";base64")) {
        const binary = atob(body);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
        return bytes.buffer;
      }
      return new TextEncoder().encode(decodeURIComponent(body)).buffer;
    }
    const response = await fetch(dataUrl);
    if (!response.ok) throw new Error(`audio fetch failed: ${response.status}`);
    return response.arrayBuffer();
  }

  hasPendingPlayback() { return this.state.audioPlaying || this.state.audioQueue.length > 0; }

  // ═══════════════  Helpers  ═══════════════

  closeWs() { if (this.state.ws && this.state.ws.readyState < WebSocket.CLOSING) { this.state.ws.close(); } }
  closeSocket(socket) {
    if (socket && socket.readyState === WebSocket.OPEN) socket.close();
    if (this.state.ws === socket) this.state.ws = null;
  }

  stopInputDevices() {
    this.state._sendingAudio = false;
    if (this.state.ws && this.state.ws.readyState === WebSocket.OPEN) {
      this.state.ws.send(JSON.stringify({ type: "stop" }));
    }
    if (this.state.timer) clearInterval(this.state.timer);
    this.state.ws = null;
    this.state.timer = null;
  }

  async flushInput() {
    if (!this.state._persistentMic || !this.state._persistentMic.node) return;
    await new Promise((resolve) => {
      const timer = setTimeout(() => { if (this.state.flushResolve === done) this.state.flushResolve = null; resolve(); }, 250);
      const done = () => { clearTimeout(timer); resolve(); };
      this.state.flushResolve = done;
      this.state._persistentMic.node.port.postMessage({ type: "flush" });
    });
  }

  resolveFlush() {
    if (!this.state.flushResolve) return;
    const resolve = this.state.flushResolve;
    this.state.flushResolve = null;
    resolve();
  }

  // ═══════════════  Server events (ws/audio)  ═══════════════

  onServerEvent(event, socket) {
    if (socket !== this.state.ws) return;
    const data = JSON.parse(event.data);
    switch (data.type) {
      case "ready": this.log(`session=${data.session_id}`); break;
      case "started":
        this.log(`server started model=${data.model} api=${data.qwen_api_base} history=${data.history_messages || 0}`);
        break;
      case "chunk_received": this.log(`chunk ${data.chunk_index} received, duration=${data.duration_ms}ms`); break;
      case "prefill_ok": this.state.prefillOk += 1; this.emitMetrics(); this.log(`prefill ${data.chunk_index} ok, latency=${data.latency_ms}ms`); break;
      case "prefill_error": this.log(`prefill ${data.chunk_index || ""} error: ${data.message || data.status_code}`, "error"); break;
      case "vad_ready": this.setVoiceState("等待说话"); this.emit("vad", { event: "ready", engine: data.engine, threshold: data.threshold }); break;
      case "vad_speech_start": this.setVoiceState("正在听你说"); this.emit("vad", { event: "speech_start", speaking: true, probability: data.probability }); break;
      case "vad_speech_end":
        this.setVoiceState("检测到说完，正在推理");
        this.emit("vad", { event: data.event || "speech_end", speaking: false, probability: data.probability });
        if (!this.options.shouldAutoStopOnVad()) break;
        this.stopRecording().catch((err) => { this.log(`server vad stop error: ${err.message || err}`, "error"); this.setMode("idle"); });
        break;
      case "vad_error": this.log(`vad error: ${data.message}`, "error"); this.emit("vad", { event: "error", message: data.message }); break;
      case "finalizing":
        this.log(`finalizing ${data.chunks} chunks`);
        if (data.easy_turn_enabled) {
          this.emit("result", { text: "判断是否说完。", replace: true });
          this.setVoiceState("判断是否说完");
        } else {
          this.emit("result", { text: "正在推理。", replace: true });
          this.setVoiceState("模型输出中");
        }
        break;
      case "turn_incomplete": this.handleTurnIncomplete(data); break;
      case "turn_complete": this.log(`easyturn ${data.turn_state} latency=${data.latency_ms}ms`); break;
      case "turn_error": this.log(`easyturn error: ${data.message || ""}`, "error"); break;
      case "final_text_delta":
        if (!this.state.streamingTextStarted) { this.emit("result", { text: "", replace: true }); this.state.streamingTextStarted = true; }
        this.emit("result", { text: data.text || "", append: true }); break;
      case "final_audio_delta":
        this.state.streamedAudioCount += 1;
        this.enqueueAudio(data.audio_data_url, data.audio_index || this.state.streamedAudioCount, data.history_text || data.speech_text || "");
        if (this.state.mode === "finalizing") this.setMode("speaking");
        this.setVoiceState("正在播报中"); break;
      case "final_result": this.handleFinalResult(data, socket); break;
      case "final_error": case "error":
        this.emit("result", { text: data.message || "请求失败。", replace: true });
        this.setVoiceState("倾听中"); this.setMode("idle");
        this.log(`${data.type}: ${data.message || data.status_code}`, "error");
        this.closeSocket(socket); break;
      default: this.log(JSON.stringify(data));
    }
  }

  handleFinalResult(data, socket) {
    this.setStatus("完成", true);
    this.state.finalReceived = true;
    this.state.finalUserText = typeof data.user_history_text === "string" ? data.user_history_text.trim() : "";
    const userDisplayText = typeof data.user_display_text === "string" ? data.user_display_text.trim() : "";
    if (userDisplayText) this.emit("userTranscript", { text: userDisplayText });
    const currentText = this.getDisplayedText();
    if (data.text) {
      this.state.finalText = data.text;
      this.state.finalHistoryText = typeof data.history_text === "string" ? data.history_text : data.text;
      this.emit("result", { text: data.text, replace: true });
    } else if (!this.state.streamingTextStarted) {
      this.emit("result", { text: "服务端没有返回文本。", replace: true });
      this.state.finalText = ""; this.state.finalHistoryText = "";
    } else {
      this.state.finalText = currentText;
      this.state.finalHistoryText = typeof data.history_text === "string" ? data.history_text : currentText;
    }
    if (data.audio_data_url) {
      this.enqueueAudio(data.audio_data_url, this.state.streamedAudioCount + 1, this.state.finalHistoryText);
    } else if (this.state.streamedAudioCount === 0) {
      this.setVoiceState("播报完");
      this.log("没有解析到音频输出。");
    }
    if (this.hasPendingPlayback()) {
      this.setMode("speaking");
    } else {
      if (this.options.shouldAutoStartOnVad()) this.clearPassiveBufferAndResetVad();
      this.setMode("idle");
    }
    this.log(`final latency=${data.latency_ms}ms, audio_chunks=${data.audio_chunks || this.state.streamedAudioCount}, input=${data.saved_input_wav}`);
    this.closeSocket(socket);
  }

  handleTurnIncomplete(data) {
    this.log(`easyturn incomplete latency=${data.latency_ms}ms audio=${Number(data.audio_seconds || 0).toFixed(2)}s`);
    const text = data.text || "嗯，我在听，你继续。";
    this.emit("result", { text, replace: true });
    this.setStatus("继续倾听", true);
    this.setVoiceState("请继续说");
    this.state.resumeAfterTurnAck = true;
    if (data.audio_data_url) {
      this.enqueueAudio(data.audio_data_url, this.state.streamedAudioCount + 1, text);
      this.setMode("speaking");
    } else {
      this.resumeRecordingAfterTurnAck();
    }
  }

  resumeRecordingAfterTurnAck() {
    if (!this.state.ws || this.state.ws.readyState !== WebSocket.OPEN) {
      this.setMode("idle");
      return;
    }
    this.state.resumeAfterTurnAck = false;

    if (this.state._persistentVadWs && this.state._persistentVadWs.readyState === WebSocket.OPEN) {
      const vadConfig = this.options.getVadConfig() || {
        engine: "silero", threshold: 0.8, minSpeechMs: 128, minSilenceMs: 800, maxSpeechMs: 30000, speechPadMs: 30,
      };
      this.state._persistentVadWs.send(JSON.stringify({
        type: "reset_vad",
        sampleRate: this.state._persistentMicSampleRate || 48000,
        vad: vadConfig,
      }));
    }

    this._drainRingBuffer();
    const preChunks = this.state.preBufferChunks;
    this.state.preBufferChunks = null;
    this.state.preBufferSampleRate = null;
    if (preChunks && preChunks.length) {
      for (let i = 0; i < preChunks.length; i += 1) {
        this.state.ws.send(preChunks[i]);
      }
      this.state.chunks += preChunks.length;
      this.log(`sent ${preChunks.length} pre-buffer chunks after turn-incomplete ack`);
    }

    this.state._sendingAudio = true;
    this.state.vad = this.createVadState();
    this.state.startedAt = performance.now();
    if (!this.state.timer) {
      this.state.timer = setInterval(() => this.emitMetrics(), 100);
    }
    this.setStatus("录音中", true);
    this.setVoiceState("正在听你说");
    this.setMode("recording");
  }

}

function defaultWebSocketUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/audio`;
}

function defaultVadMonitorWebSocketUrl() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/vad`;
}
