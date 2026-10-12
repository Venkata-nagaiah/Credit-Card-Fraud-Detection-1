/**
 * Web Audio API Sound Synthesizer for Credit Card Fraud Detection
 * Zero external audio files required. Generates high-urgency security buzzer for fraud
 * and harmonic verification chime for legitimate transactions.
 */

class FraudAudioAlert {
  constructor() {
    this.audioCtx = null;
    this.muted = false;
  }

  init() {
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (AudioContext) {
        this.audioCtx = new AudioContext();
      }
    }
    if (this.audioCtx && this.audioCtx.state === "suspended") {
      this.audioCtx.resume();
    }
  }

  toggleMute() {
    this.muted = !this.muted;
    return this.muted;
  }

  playFraudBuzzer() {
    if (this.muted) return;
    this.init();
    if (!this.audioCtx) return;

    const ctx = this.audioCtx;
    const now = ctx.currentTime;

    // Pulse 1
    this._playBeep(ctx, now, 0.16, 750, "sawtooth");
    this._playBeep(ctx, now, 0.16, 375, "square");

    // Pulse 2
    this._playBeep(ctx, now + 0.22, 0.16, 750, "sawtooth");
    this._playBeep(ctx, now + 0.22, 0.16, 375, "square");

    // Pulse 3 (sustained alert)
    this._playBeep(ctx, now + 0.44, 0.35, 880, "sawtooth");
    this._playBeep(ctx, now + 0.44, 0.35, 440, "square");
  }

  playSafeChime() {
    if (this.muted) return;
    this.init();
    if (!this.audioCtx) return;

    const ctx = this.audioCtx;
    const now = ctx.currentTime;

    // Harmonic pleasant chord (C5 -> E5 -> G5)
    this._playChimeTone(ctx, now, 0.35, 523.25);
    this._playChimeTone(ctx, now + 0.12, 0.40, 659.25);
    this._playChimeTone(ctx, now + 0.24, 0.55, 783.99);
  }

  _playBeep(ctx, startTime, duration, freq, type = "sawtooth") {
    try {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, startTime);

      gain.gain.setValueAtTime(0.001, startTime);
      gain.gain.exponentialRampToValueAtTime(0.28, startTime + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(startTime);
      osc.stop(startTime + duration);
    } catch (e) {
      console.warn("Audio synthesizer error:", e);
    }
  }

  _playChimeTone(ctx, startTime, duration, freq) {
    try {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sine";
      osc.frequency.setValueAtTime(freq, startTime);

      gain.gain.setValueAtTime(0.001, startTime);
      gain.gain.exponentialRampToValueAtTime(0.18, startTime + 0.04);
      gain.gain.exponentialRampToValueAtTime(0.001, startTime + duration);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(startTime);
      osc.stop(startTime + duration);
    } catch (e) {
      console.warn("Audio synthesizer error:", e);
    }
  }
}

window.fraudAudio = new FraudAudioAlert();
