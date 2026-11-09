import { HttpClient } from '@angular/common/http';
import { inject, Injectable, signal } from '@angular/core';
import { forkJoin, map, Observable, tap } from 'rxjs';
import { Recording } from '../interfaces/recording';

@Injectable({
  providedIn: 'root'
})
export class BackendService {
  private http: HttpClient = inject(HttpClient);
  url = signal<string>('https://ssi-gatherer.pb.ax');

  text_distribution = signal<Map<string, number>>(new Map<string, number>());

  readonly health = '/health';
  readonly save = '/recordings';
  readonly recordings = '/recordings';
  readonly text = '/text';
  readonly analyze = '/analyze';

  constructor() {
    forkJoin({
      recordings: this.getRecordings(),
      texts: this.getText()
    }).subscribe(({ recordings, texts }) => {
      texts.forEach((text) => {
        this.text_distribution().set(text[1], 0);
      });
      recordings.forEach((recording) => {
        const count = this.text_distribution().get(recording.text)!;
        this.text_distribution().set(recording.text, count + 1);
      });

    });
  }

  healthCheck(url?: string): Observable<any> {
    const uri = (url || this.url()).concat(this.health);
    return this.http.get(uri);
  }

  saveRecording(recording: Recording, url?: string): Observable<any> {
    const uri = (url || this.url()).concat(this.save);
    const formData = new FormData();
    formData.append('unvoiced', recording.unvoiced!);
    formData.append('voiced', recording.voiced!);
    formData.append('text', recording.text);
    formData.append('uuid', recording.uuid);
    formData.append('timestamp', recording.timestamp);
    return this.http.post(uri, formData);
  }

  saveText(text: string, url?: string): Observable<any> {
    const uri = (url || this.url()).concat(this.text);
    return this.http.post(uri, { text: text }).pipe(
      tap(() => this.text_distribution().set(text, 0))
    );
  }

  getRecordings(url?: string): Observable<any[]> {
    const uri = (url || this.url()).concat(this.recordings);
    return this.http.get<any[]>(uri);
  }

  getText(url?: string): Observable<any[]> {
    const uri = (url || this.url()).concat(this.text);
    return this.http.get<any[]>(uri);
  }
}
