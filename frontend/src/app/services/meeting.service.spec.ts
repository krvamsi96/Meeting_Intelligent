import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { MeetingService } from './meeting.service';
import { environment } from '../../environments/environment';
import { FinalResult } from '../models/api.models';

describe('MeetingService', () => {
  let service: MeetingService;
  let http: HttpTestingController;
  const api = environment.apiUrl;

  beforeEach(() => {
    TestBed.configureTestingModule({ providers: [provideHttpClient(), provideHttpClientTesting()] });
    service = TestBed.inject(MeetingService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('asks the list endpoint for a page with the filters', () => {
    service.listMeetings({ page: 2, limit: 10, status: 'failed', search: ' budget ', sort: 'oldest' }).subscribe();
    const req = http.expectOne((r) => r.url === `${api}/meetings`);
    expect(req.request.params.get('page')).toBe('2');
    expect(req.request.params.get('status')).toBe('failed');
    expect(req.request.params.get('search')).toBe('budget');
    expect(req.request.params.get('sort')).toBe('oldest');
    req.flush({ items: [], total: 0, page: 2, limit: 10, pages: 1, status_counts: { completed: 0, processing: 0, failed: 0 } });
  });

  it('treats the 202 answer of /final as "results not ready", not as an empty completed meeting', () => {
    let result: FinalResult | undefined;
    service.getMeeting(7).subscribe((r) => (result = r));
    http.expectOne(`${api}/meetings/7/final`).flush(
      { processing_status: 'PENDING', status: 'processing', detail: {} },
      { status: 202, statusText: 'Accepted' },
    );
    expect(result?.kind).toBe('pending');
  });

  it('returns the meeting when /final has results', () => {
    let result: FinalResult | undefined;
    service.getMeeting(7).subscribe((r) => (result = r));
    http.expectOne(`${api}/meetings/7/final`).flush({ processing_status: 'COMPLETED', meeting: { meeting_pk: 7 }, transcript: [] });
    expect(result?.kind).toBe('ready');
  });

  it('sends the title field the upload endpoint reads, and the language only when chosen', () => {
    const file = new File(['x'], 'standup.mp4', { type: 'video/mp4' });
    service.upload(file, ' Weekly standup ', null).subscribe();
    const req = http.expectOne(`${api}/upload`);
    const body = req.request.body as FormData;
    expect(body.get('title')).toBe('Weekly standup');
    expect(body.has('language')).toBe(false);
    req.flush({ id: 1, meeting_id: '1', status: 'processing' });
  });

  it('uses the real chat and search routes', () => {
    service.ask(5, 'What was decided?', null).subscribe();
    const chat = http.expectOne(`${api}/meetings/5/chat`);
    expect(chat.request.body).toEqual({ message: 'What was decided?' });
    chat.flush({ content: 'ok' });

    service.search(' budget ').subscribe();
    const search = http.expectOne((r) => r.url === `${api}/search`);
    expect(search.request.params.get('q')).toBe('budget');
    search.flush({ query: 'budget', results: [], total: 0 });
  });
});
