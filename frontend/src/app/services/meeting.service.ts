import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpEvent, HttpParams, HttpRequest } from '@angular/common/http';
import { Observable, map } from 'rxjs';
import { environment } from '../../environments/environment';
import {
  ChatReply,
  FinalMeeting,
  FinalPending,
  FinalResult,
  MeetingListPage,
  MeetingListQuery,
  ProcessingStatus,
  SearchResponse,
  ServiceStatus,
  UploadOptions,
  UploadResult,
} from '../models/api.models';

/** Every call the user-facing screens make. Paths and payloads match the backend routes exactly. */
@Injectable({ providedIn: 'root' })
export class MeetingService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  listMeetings(query: MeetingListQuery): Observable<MeetingListPage> {
    let params = new HttpParams()
      .set('page', String(query.page ?? 1))
      .set('limit', String(query.limit ?? 10))
      .set('sort', query.sort ?? 'newest');
    if (query.status) params = params.set('status', query.status);
    if (query.search?.trim()) params = params.set('search', query.search.trim());
    return this.http.get<MeetingListPage>(`${this.api}/meetings`, { params });
  }

  /** Results of one meeting; `pending` while they are not ready yet (HTTP 202). */
  getMeeting(id: number | string): Observable<FinalResult> {
    return this.http
      .get<FinalMeeting | FinalPending>(`${this.api}/meetings/${encodeURIComponent(String(id))}/final`, { observe: 'response' })
      .pipe(
        map((res): FinalResult =>
          res.status === 202 || (res.body as FinalPending | null)?.processing_status === 'PENDING'
            ? { kind: 'pending', pending: res.body as FinalPending }
            : { kind: 'ready', meeting: res.body as FinalMeeting },
        ),
      );
  }

  getStatus(id: number | string): Observable<ProcessingStatus> {
    return this.http.get<ProcessingStatus>(`${this.api}/meetings/${encodeURIComponent(String(id))}/status`);
  }

  getUploadOptions(): Observable<UploadOptions> {
    return this.http.get<UploadOptions>(`${this.api}/upload/options`);
  }

  upload(file: File, title: string, language: string | null): Observable<HttpEvent<UploadResult>> {
    const form = new FormData();
    form.append('file', file);
    if (title.trim()) form.append('title', title.trim());
    if (language) form.append('language', language);
    return this.http.request<UploadResult>(new HttpRequest('POST', `${this.api}/upload`, form, { reportProgress: true }));
  }

  deleteMeeting(id: number): Observable<unknown> {
    return this.http.delete(`${this.api}/meetings/${id}`);
  }

  retry(id: number | string): Observable<unknown> {
    return this.http.post(`${this.api}/meetings/${encodeURIComponent(String(id))}/retry`, {});
  }

  search(query: string, limit = 20): Observable<SearchResponse> {
    const params = new HttpParams().set('q', query.trim()).set('limit', String(limit));
    return this.http.get<SearchResponse>(`${this.api}/search`, { params });
  }

  ask(meetingId: number | string, message: string, sessionId: string | null): Observable<ChatReply> {
    return this.http.post<ChatReply>(`${this.api}/meetings/${encodeURIComponent(String(meetingId))}/chat`, {
      message,
      ...(sessionId ? { session_id: sessionId } : {}),
    });
  }

  download(meetingId: number | string, format: 'docx' | 'xlsx'): Observable<Blob> {
    return this.http.get(`${this.api}/meetings/${encodeURIComponent(String(meetingId))}/export/${format}`, { responseType: 'blob' });
  }

  /** Address of the printable report (opened in a new tab; the browser's Print dialog saves it as PDF). */
  printableReportUrl(meetingId: number | string): string {
    return `${this.api}/meetings/${encodeURIComponent(String(meetingId))}/export/pdf`;
  }

  serviceStatus(): Observable<ServiceStatus> {
    return this.http.get<ServiceStatus>(`${this.api}/system/status`);
  }
}
