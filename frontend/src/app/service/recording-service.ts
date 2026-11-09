import { inject, Injectable, signal } from '@angular/core';
import { Recording } from '../interfaces/recording';
import { BackendService } from './backend';
import { v4 as uuidv4 } from 'uuid';

@Injectable({
  providedIn: 'root'
})
export class RecordingService {
  // Injections
  backend = inject(BackendService);

  // Variables
  osc: OscillatorNode | null = null;
  gainNode: GainNode | null = null;
  audioContext: AudioContext | null = null;
  analyserNode: AnalyserNode | null = null;

  // Signals
  currentRecording = signal<Recording | null>(null);
  isRecording = signal<boolean>(false);
  chunks = signal<BlobPart[]>([]);
  stream = signal<MediaStream | null>(null);
  mediaRecorder = signal<MediaRecorder | null>(null);
  text = signal<string>('');

  // Dynamic gain signals
  carrierGain = signal<number>(1.0);
  agcActive = signal<boolean>(false);
  carrierLevel = signal<number>(0);// Normalised carrier level (0-1)
  agcStatus = signal<string>('');

  private animationFrameId: number | null = null;

  // Mono downmix graph (kept referenced so it can be cleaned up after each recording)
  private monoSource: MediaStreamAudioSourceNode | null = null;
  private monoDestination: MediaStreamAudioDestinationNode | null = null;

  createNewRecording() {
    this.backend.getText().subscribe((texts: any[]) => {
      const textsMap = new Map<string, number>();
      this.backend.text_distribution().forEach((count, text) => textsMap.set(text, count));
      const minCount = Math.min(...textsMap.values());
      const textsWithMinCount = [...textsMap.entries()].filter(([_, count]) => count === minCount);
      const text = textsWithMinCount[Math.floor(Math.random() * textsWithMinCount.length)][0];
      console.log(text, textsWithMinCount);
      this.text.set(text);
      this.currentRecording()!.text = text;
    })
    const recording: Recording = {
      uuid: uuidv4(),
      text: this.text(),
      timestamp: new Date().toISOString(),
    };
    this.currentRecording.set(recording);
  }

  startCW(gain?: number) {
    if (this.audioContext == null) {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    const gainValue = gain ?? this.carrierGain();

    this.osc = this.audioContext.createOscillator();
    this.osc.type = 'sine';
    this.osc.frequency.value = 19000;

    this.gainNode = this.audioContext.createGain();
    this.gainNode.gain.value = gainValue;

    this.osc.connect(this.gainNode);
    this.gainNode.connect(this.audioContext.destination);
    this.osc.start();

    // Start carrier level monitoring
    this.startCarrierLevelMonitor();
  }

  stopCW() {
    this.stopCarrierLevelMonitor();
    if (this.osc) {
      this.osc.stop();
      this.osc = null;
    }
    this.gainNode = null;
  }

  /**
   * Run Automatic Gain Control:
   * 1. Emit a short CW pulse at default gain
   * 2. Measure received level via AnalyserNode
   * 3. Compute optimal gain
   * 4. Set the GainNode accordingly
   */
  async runAGC(): Promise<number> {
    const targetLevel = 0.3;  // Target normalised carrier level
    const testGain = 1.0;
    const measureDuration = 300; // ms

    this.agcStatus.set('AGC: measuring...');
    this.agcActive.set(true);

    if (this.audioContext == null) {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    }

    // Create test oscillator + gain
    const testOsc = this.audioContext.createOscillator();
    testOsc.type = 'sine';
    testOsc.frequency.value = 19000;

    const testGainNode = this.audioContext.createGain();
    testGainNode.gain.value = testGain;

    testOsc.connect(testGainNode);
    testGainNode.connect(this.audioContext.destination);

    // Set up analyser to measure 18-20 kHz band
    const analyser = this.audioContext.createAnalyser();
    analyser.fftSize = 2048;

    // Connect microphone input to analyser if stream is available
    let micSource: MediaStreamAudioSourceNode | null = null;
    if (this.stream()) {
      micSource = this.audioContext.createMediaStreamSource(this.stream()!);
      micSource.connect(analyser);
    }

    testOsc.start();

    // Measure for measureDuration ms
    await new Promise<void>((resolve) => {
      setTimeout(() => {
        testOsc.stop();
        if (micSource) {
          micSource.disconnect();
        }
        testGainNode.disconnect();
        resolve();
      }, measureDuration);
    });

    // Read analyser data for carrier band
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    analyser.getByteFrequencyData(dataArray);

    // Estimate energy in 18-20 kHz band
    // At 44100 Hz with fftSize=2048, each bin is ~21.5 Hz
    // 18-20 kHz is bins ~837-930
    const binStart = Math.floor(18000 / (this.audioContext.sampleRate / bufferLength));
    const binEnd = Math.floor(20000 / (this.audioContext.sampleRate / bufferLength));
    let carrierEnergy = 0;
    let count = 0;
    for (let i = binStart; i <= binEnd && i < bufferLength; i++) {
      carrierEnergy += dataArray[i] * dataArray[i];
      count++;
    }
    const measuredLevel = Math.sqrt(carrierEnergy / Math.max(count, 1)) / 255;

    // Compute required gain
    let optimalGain = testGain;
    if (measuredLevel > 0.01) {
      optimalGain = Math.min(Math.max((targetLevel / measuredLevel) * testGain, 0.1), 5.0);
    }

    this.carrierGain.set(parseFloat(optimalGain.toFixed(2)));
    this.agcStatus.set(`AGC: gain set to ${optimalGain.toFixed(1)}x`);
    this.agcActive.set(false);

    // Clean up analyser
    analyser.disconnect();

    return optimalGain;
  }

  /**
   * Start real-time carrier level monitoring in the 18-20 kHz band.
   */
  private startCarrierLevelMonitor() {
    if (!this.audioContext || !this.stream()) return;

    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 2048;

    const micSource = this.audioContext.createMediaStreamSource(this.stream()!);
    micSource.connect(this.analyserNode);

    const bufferLength = this.analyserNode.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    const sampleRate = this.audioContext.sampleRate;
    const binFreq = sampleRate / bufferLength;
    const binStart = Math.floor(18000 / binFreq);
    const binEnd = Math.floor(20000 / binFreq);

    const update = () => {
      if (!this.analyserNode) return;
      this.analyserNode.getByteFrequencyData(dataArray);

      let energy = 0;
      let count = 0;
      for (let i = binStart; i <= binEnd && i < bufferLength; i++) {
        energy += dataArray[i] * dataArray[i];
        count++;
      }
      const level = Math.sqrt(energy / Math.max(count, 1)) / 255;
      this.carrierLevel.set(level);

      this.animationFrameId = requestAnimationFrame(update);
    };

    update();
  }

  private stopCarrierLevelMonitor() {
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    this.analyserNode = null;
  }

  /**
   * Returns a mono version of the given audio stream.
   *
   * 1. Asks the browser/device for a single channel via track constraints.
   * 2. Downmixes the stream to one channel through the Web Audio API, so the
   *    recording is guaranteed to be mono even if the input device (e.g. a
   *    large membrane mic) delivers a stereo signal.
   */
  private forceMono(stream: MediaStream): MediaStream {
    // Fast path: request a single channel directly from the device.
    const audioTrack = stream.getAudioTracks()[0];
    if (audioTrack) {
      try {
        audioTrack.applyConstraints({ channelCount: 1 });
      } catch {
        // Some devices/browsers ignore channelCount; the downmix below still guarantees mono.
      }
    }

    // Guarantee mono by downmixing through the Web Audio API.
    if (this.audioContext == null) {
      this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    }
    if (this.audioContext.state === 'suspended') {
      this.audioContext.resume();
    }

    this.monoSource = this.audioContext.createMediaStreamSource(stream);
    this.monoDestination = this.audioContext.createMediaStreamDestination();
    this.monoDestination.channelCount = 1;
    this.monoDestination.channelCountMode = 'explicit';
    this.monoDestination.channelInterpretation = 'speakers';

    this.monoSource.connect(this.monoDestination);

    return this.monoDestination.stream;
  }

  /** Releases the mono downmix graph after a recording has finished. */
  private releaseMonoGraph() {
    if (this.monoSource) {
      this.monoSource.disconnect();
      this.monoSource = null;
    }
    if (this.monoDestination) {
      this.monoDestination.disconnect();
      this.monoDestination = null;
    }
  }

  startRecording(type: 'voiced' | 'unvoiced') {
    if (this.stream() == null) {
      throw new Error('Stream is null and needs to be set in the component.');
    }
    this.chunks.set([]);
    this.isRecording.set(true);

    const recordingStream = this.forceMono(this.stream()!);
    const recorder = new MediaRecorder(recordingStream, { mimeType: 'audio/webm' });
    const recordingType = type; // Capture for the closure

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.chunks.update(c => [...c, event.data]);
      }
    };

    recorder.onstop = () => {
      const blob = new Blob(this.chunks(), { type: 'audio/webm' });
      if (recordingType === 'voiced') {
        this.currentRecording()!.voiced = blob;
      } else {
        this.currentRecording()!.unvoiced = blob;
      }
      this.chunks.set([]);
      this.isRecording.set(false);
      this.releaseMonoGraph();
    };

    recorder.start();
    this.mediaRecorder.set(recorder);
  }

  stopRecording() {
    this.mediaRecorder()?.stop();
    this.isRecording.set(false);
  }

}
