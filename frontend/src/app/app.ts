import { Component, inject, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { Recording } from './interfaces/recording';
import { BackendService } from './service/backend';
import { RecordingService } from './service/recording-service';
import { Navbar } from './components/navbar/navbar';

@Component({
  selector: 'app-root',
  templateUrl: './app.html',
  styleUrl: './app.css',
  imports: [Navbar, RouterOutlet]
})
export class App {
  // Injections
  backend = inject(BackendService);
  recordingService = inject(RecordingService);

  protected readonly title = signal('ssi-gatherer');
  mediaDevices = signal<MediaDeviceInfo[]>([]);
  recordings = signal<Recording[]>([]);

  /**
   * Get a list of all recordings from the server.
   *
   * This function queries the server for a list of all recordings and updates
   * the `recordings` signal with the received list.
   */
  getRecordings() {
    this.backend.getRecordings().subscribe((recordings) => {
      this.recordings.set(recordings);
    });
  }

  /**
   * Save a recording to the server.
   *
   * This function sends the recording to the server and logs the response
   * to the console. If the server responds with an error, the error is
   * also logged to the console.
   *
   * @param recording The recording to be saved.
   */
  saveRecording(recording: Recording) {
    this.backend.saveRecording(recording).subscribe((response) => {
      console.log(response);
    });
  }

  /**
   * Prepare a new recording.
   *
   * This function prepares a new recording by creating a new recording object
   * and clearing the text input field. It does not start the recording.
   */
  setupRecording() {
    this.recordingService.createNewRecording();
  }

  /**
   * Handle voiced recording toggle.
   *
   * This function checks if a recording is currently in progress. If it is,
   * the function stops both the continuous wave (CW) and the recording.
   * If no recording is in progress, it starts the CW and begins a new
   * 'voiced' recording.
   */
  handleVoiced() {
    if (this.recordingService.isRecording()) {
      this.recordingService.stopCW();
      this.recordingService.stopRecording();
    } else {
      this.recordingService.startCW();
      this.recordingService.startRecording('voiced');
    }
  }

  /**
   * Handle unvoiced recording toggle.
   *
   * This function checks if a recording is currently in progress. If it is,
   * the function stops both the continuous wave (CW) and the recording.
   * If no recording is in progress, it starts the CW and begins a new
   * 'unvoiced' recording.
   */
  handleUnvoiced() {
    if (this.recordingService.isRecording()) {
      this.recordingService.stopCW();
      this.recordingService.stopRecording();
    } else {
      this.recordingService.startCW();
      this.recordingService.startRecording('unvoiced');
    }
  }

}
