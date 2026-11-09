import { Component, inject, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormControl, FormGroup, FormsModule, ReactiveFormsModule, Validators } from '@angular/forms';
import { BackendService } from '../../service/backend';
import { ConnectionStatus } from '../../interfaces/connection-status';
import { RecordingService } from '../../service/recording-service';
import { catchError } from 'rxjs/operators';

@Component({
  selector: 'app-recording',
  imports: [ReactiveFormsModule, FormsModule, CommonModule],
  templateUrl: './recording.html',
  styleUrl: './recording.css'
})
export class Recording implements OnInit {
  backendService = inject(BackendService);
  recordingService = inject(RecordingService);

  urlControl = new FormControl<string>('', [Validators.required]);
  formGroup = new FormGroup({
    url: this.urlControl
  });
  voicedPressed = false;
  unvoicedPressed = false;

  connectionStatus = signal<ConnectionStatus>(ConnectionStatus.Unknown);
  mediaDevices = signal<MediaDeviceInfo[]>([]);
  permissionStatus = signal<PermissionState>('prompt');

  ngOnInit(): void {
    this.formGroup.controls.url.setValue(this.backendService.url());
  }

  checkConnection(): void {
    if (this.formGroup.invalid) {
      return;
    }
    const value = this.formGroup.controls.url.value!;
    console.log(value);
    this.backendService.url.set(value);
    this.backendService.healthCheck(value).pipe(
      catchError((error) => {
        this.connectionStatus.set(ConnectionStatus.Disconnected);
        return error;
      })
    ).subscribe(
      (response) => {
        this.connectionStatus.set(ConnectionStatus.Connected);
      }
    )
  }

  askForMicrophone(): void {
    navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        sampleRate: 44100
      }
    })
      .then((stream) => {
        this.recordingService.stream.set(stream);
        this.permissionStatus.set('granted');
      })
      .catch((error) => {
        console.error(error);
        this.permissionStatus.set('denied');
      });
  }

  listMediaDevices() {
    navigator.mediaDevices.enumerateDevices()
      .then((devices) => {
        console.log(devices);
        this.mediaDevices.set(devices);
      })
      .catch((error) => {
        console.error(error);
      });
  }

  createNewRecording() {
    this.recordingService.createNewRecording();
  }

  printRecording() {
    console.log(this.recordingService.currentRecording());
  }

  recordVoiced() {
    if (this.voicedPressed) {
      this.recordingService.stopRecording();
      this.recordingService.stopCW();
      this.voicedPressed = false;
      return;
    }
    this.voicedPressed = true;
    this.recordingService.startCW(this.recordingService.carrierGain());
    this.recordingService.startRecording('voiced');
  }

  recordUnvoiced() {
    if (this.unvoicedPressed) {
      this.recordingService.stopRecording();
      this.recordingService.stopCW();
      this.unvoicedPressed = false;
      return;
    }
    this.unvoicedPressed = true;
    this.recordingService.startCW(this.recordingService.carrierGain());
    this.recordingService.startRecording('unvoiced');
  }

  async runAGC() {
    await this.recordingService.runAGC();
  }

  setGain(event: Event) {
    const value = parseFloat((event.target as HTMLInputElement).value);
    this.recordingService.carrierGain.set(value);
  }

  saveRecording() {
    if (this.recordingService.currentRecording() == null) {
      return;
    }
    this.backendService.saveRecording(this.recordingService.currentRecording()!).subscribe(
      (response) => {
        console.log(response);
      }
    );
  }

}
