class PCMChunker extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const processorOptions = options.processorOptions || {};
    this.chunkMs = Number(processorOptions.chunkMs || 600);
    this.chunkSamples = Math.max(128, Math.round((sampleRate * this.chunkMs) / 1000));
    this.pending = [];
    this.pendingSamples = 0;

    this.port.onmessage = (event) => {
      if (event.data && event.data.type === "flush") {
        this.flushPending();
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || input[0].length === 0) {
      return true;
    }

    const channels = input.length;
    const frameCount = input[0].length;
    const mono = new Float32Array(frameCount);

    if (channels === 1) {
      mono.set(input[0]);
    } else {
      for (let i = 0; i < frameCount; i += 1) {
        let sum = 0;
        for (let ch = 0; ch < channels; ch += 1) {
          sum += input[ch][i] || 0;
        }
        mono[i] = sum / channels;
      }
    }

    this.pending.push(mono);
    this.pendingSamples += mono.length;

    while (this.pendingSamples >= this.chunkSamples) {
      const chunk = new Float32Array(this.chunkSamples);
      let offset = 0;

      while (offset < this.chunkSamples && this.pending.length > 0) {
        const head = this.pending[0];
        const need = this.chunkSamples - offset;
        if (head.length <= need) {
          chunk.set(head, offset);
          offset += head.length;
          this.pending.shift();
        } else {
          chunk.set(head.subarray(0, need), offset);
          this.pending[0] = head.subarray(need);
          offset += need;
        }
      }

      this.pendingSamples -= this.chunkSamples;
      this.port.postMessage({ type: "chunk", pcm: chunk.buffer }, [chunk.buffer]);
    }

    return true;
  }

  flushPending() {
    if (this.pendingSamples > 0) {
      const chunk = new Float32Array(this.pendingSamples);
      let offset = 0;
      while (this.pending.length > 0) {
        const head = this.pending.shift();
        chunk.set(head, offset);
        offset += head.length;
      }
      this.pendingSamples = 0;
      this.port.postMessage({ type: "chunk", pcm: chunk.buffer, flush: true }, [chunk.buffer]);
    }
    this.port.postMessage({ type: "flushed" });
  }
}

registerProcessor("pcm-chunker", PCMChunker);
