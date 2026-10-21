import { TestBed, ComponentFixture } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { HttpTestingController, provideHttpClientTesting } from '@angular/common/http/testing';
import { provideRouter } from '@angular/router';
import { Type } from '@angular/core';
import { environment } from '../../environments/environment';
import { DashboardComponent } from './dashboard/dashboard.component';
import { MeetingDetailComponent } from './meeting-detail/meeting-detail.component';
import { MeetingListComponent } from './meeting-list/meeting-list.component';
import { ProcessingStatusComponent } from './processing-status/processing-status.component';
import { UploadComponent, validateFile } from './upload/upload.component';
import { SearchComponent } from './search/search.component';
import { AppComponent } from '../app.component';
import { UploadOptions } from '../models/api.models';

const api = environment.apiUrl;

/** Internal technology / provider / infrastructure words that must never appear on an ordinary user's screen. */
const INTERNAL = /airawat|groq|gemini|ollama|openai|llama|whisper|nemo|chroma|postgres|sql|fastapi|python|ffmpeg|angular|vector|embedding|diariz|traceback|exception|localhost|api\/v1|model id/i;

function setup<T>(component: Type<T>, inputs: Record<string, unknown> = {}) {
  TestBed.configureTestingModule({
    imports: [component],
    providers: [provideHttpClient(), provideHttpClientTesting(), provideRouter([])],
  });
  const fixture = TestBed.createComponent(component);
  for (const [k, v] of Object.entries(inputs)) fixture.componentRef.setInput(k, v);
  return { fixture, http: TestBed.inject(HttpTestingController) };
}

async function render<T>(fixture: ComponentFixture<T>): Promise<string> {
  fixture.detectChanges();
  await fixture.whenStable();
  fixture.detectChanges();
  return (fixture.nativeElement as HTMLElement).textContent ?? '';
}

const counts = { completed: 7, processing: 2, failed: 1 };
const listPage = (items: unknown[]) => ({ items, total: items.length, page: 1, limit: 10, pages: 1, status_counts: counts });
const row = {
  id: 12, meeting_id: '12', title: 'Quarterly planning', filename: 'q3.mp4', file_type: 'media', status: 'failed',
  duration_seconds: 1830, speaker_count: 3, language: 'en', uploaded_at: '2026-10-20T09:00:00Z',
  error: { code: 'AI_SERVICE_UNAVAILABLE', message: 'The AI service is temporarily unavailable. Please try again later.' },
};

describe('Dashboard', () => {
  it('shows totals for ALL meetings from the server, not counts of the rows on screen', async () => {
    const { fixture, http } = setup(DashboardComponent);
    fixture.detectChanges();
    http.expectOne((r) => r.url === `${api}/meetings`).flush(listPage([row]));
    const text = await render(fixture);
    const values = [...(fixture.nativeElement as HTMLElement).querySelectorAll('.stat-value')].map((e) => e.textContent?.trim());
    expect(values).toEqual(['10', '7', '2', '1']);     // 7 + 2 + 1 overall, although only one row is listed
    expect(text).toContain('Quarterly planning');
    expect(text).toContain('Needs attention');
    expect(text).not.toMatch(INTERNAL);
  });

  it('shows an honest error with a retry instead of invented data when the list cannot be loaded', async () => {
    const { fixture, http } = setup(DashboardComponent);
    fixture.detectChanges();
    http.expectOne((r) => r.url === `${api}/meetings`).flush({ message: 'psycopg2 OperationalError' }, { status: 500, statusText: 'x' });
    const text = await render(fixture);
    expect(text).toContain('Your meetings could not be loaded');
    expect(text).toContain('Try again');
    expect(text).not.toMatch(INTERNAL);
    expect(text).not.toMatch(/Completed\s*\d/);
  });
});

describe('Meeting list', () => {
  it('lists meetings with the plain-language error of a failed one and offers retry', async () => {
    const { fixture, http } = setup(MeetingListComponent);
    fixture.detectChanges();
    http.expectOne((r) => r.url === `${api}/meetings`).flush(listPage([row]));
    const text = await render(fixture);
    expect(text).toContain('Quarterly planning');
    expect(text).toContain('The AI service is temporarily unavailable');
    expect(text).toContain('Try again');
    expect(text).not.toMatch(INTERNAL);
  });
});

const finalMeeting = {
  processing_status: 'COMPLETED', display_status: 'completed', user_error: null,
  meeting: { meeting_pk: 12, title: 'Quarterly planning', meeting_date: '2026-10-20', duration_seconds: 1830, speaker_count: 2, source_language: 'en' },
  transcript: [
    { speaker: 'SPEAKER_00', start_time: '0.000', end_time: '6.5', text: 'We agreed to launch in March.' },
    { speaker: 'Speaker_2', start_time: '6.5', end_time: '12.0', text: 'I will prepare the budget by Friday.' },
  ],
  speaker_duration: [{ speaker_id: 'SPEAKER_00', duration_seconds: 900, percentage: 60, segment_count: 10 }],
  agenda: [{ agenda_id: 'A1', title: 'Launch date', order: 1 }],
  decisions: [{ agenda_id: 'A1', decision: 'Launch in March', evidence: 'We agreed to launch in March.', speaker: 'SPEAKER_00', timestamp: '0.0' }],
  agenda_decision_mapping: [],
  action_items: [{ task: 'Prepare the budget', owner: 'Unassigned', deadline: 'Friday', evidence: 'I will prepare the budget by Friday.' }],
  summary: 'The team agreed the launch date.',
  mom: '## Decisions\n- Launch in March',
  sentiment: { label: 'Positive' },
};

describe('Meeting detail', () => {
  it('renders the saved results with speaker labels, and never invents an owner', async () => {
    const { fixture, http } = setup(MeetingDetailComponent, { id: '12' });
    fixture.detectChanges();
    http.expectOne(`${api}/meetings/12/final`).flush(finalMeeting);
    let text = await render(fixture);
    expect(text).toContain('Quarterly planning');
    expect(text).toContain('The team agreed the launch date.');
    expect(text).toContain('Speaker 1');
    expect(text).not.toMatch(INTERNAL);

    const host = fixture.nativeElement as HTMLElement;
    (host.querySelector('#tab-actions') as HTMLButtonElement).click();
    text = await render(fixture);
    expect(text).toContain('Prepare the budget');
    expect(text).toContain('Not assigned');
    expect(text).not.toContain('Unassigned');

    (host.querySelector('#tab-decisions') as HTMLButtonElement).click();
    text = await render(fixture);
    expect(text).toContain('Launch in March');
    expect(text).toContain('We agreed to launch in March.');
  });

  it('shows "being prepared" - not an empty completed meeting - while results are not ready', async () => {
    const { fixture, http } = setup(MeetingDetailComponent, { id: '12' });
    fixture.detectChanges();
    http.expectOne(`${api}/meetings/12/final`).flush({ processing_status: 'PENDING', status: 'processing' }, { status: 202, statusText: 'Accepted' });
    const text = await render(fixture);
    expect(text).toContain('Your results are being prepared');
    expect(text).not.toContain('Completed');
  });

  it('says the latest run failed when only older results exist', async () => {
    const { fixture, http } = setup(MeetingDetailComponent, { id: '12' });
    fixture.detectChanges();
    http.expectOne(`${api}/meetings/12/final`).flush({
      ...finalMeeting, display_status: 'failed',
      user_error: { code: 'NO_SPEECH', message: 'No speech could be recognised in this recording.' },
    });
    const text = await render(fixture);
    expect(text).toContain('Needs attention');
    expect(text).toContain('No speech could be recognised');
    expect(text).toContain('previous successful run');
  });
});

describe('Processing status', () => {
  it('shows plain steps and the user-safe reason when processing failed', async () => {
    const { fixture, http } = setup(ProcessingStatusComponent, { id: '12' });
    fixture.detectChanges();
    http.expectOne(`${api}/meetings/12/status`).flush({
      meeting_id: 12, status: 'failed', progress_percent: 43, has_warnings: false,
      steps: [
        { key: 'upload', label: 'Uploading', state: 'completed' },
        { key: 'transcribe', label: 'Processing meeting', state: 'failed' },
        { key: 'results', label: 'Preparing results', state: 'pending' },
        { key: 'done', label: 'Needs attention', state: 'failed' },
      ],
      error: 'The AI service is temporarily unavailable. Please try again later.', error_code: 'AI_SERVICE_UNAVAILABLE',
    });
    const text = await render(fixture);
    expect(text).toContain('Processing meeting');
    expect(text).toContain('43%');
    expect(text).toContain('The AI service is temporarily unavailable');
    expect(text).not.toContain('View results');
    expect(text).not.toMatch(INTERNAL);
    http.verify();   // a failed meeting is not polled again
  });
});

describe('Upload', () => {
  const options: UploadOptions = {
    accepted_extensions: ['.mp3', '.mp4', '.wav'], max_file_size_mb: 1, language_detection: true,
    default_language: null, language_required: false, languages: [{ code: 'en', name: 'English' }],
  };

  it('rejects unsupported, empty and oversized files before uploading', () => {
    expect(validateFile(new File(['x'], 'notes.pdf'), options)).toMatch(/not supported/);
    expect(validateFile(new File([], 'empty.mp3'), options)).toMatch(/empty/);
    expect(validateFile(new File([new Uint8Array(2 * 1024 * 1024)], 'big.wav'), options)).toMatch(/largest file/);
    expect(validateFile(new File(['x'], 'ok.MP4'), options)).toBeNull();
  });

  it('builds the form from the server limits', async () => {
    const { fixture, http } = setup(UploadComponent);
    fixture.detectChanges();
    http.expectOne(`${api}/upload/options`).flush(options);
    const text = await render(fixture);
    expect(text).toContain('up to 1 MB');
    expect(text).toContain('Detect automatically');
    expect(text).not.toMatch(INTERNAL);
  });
});

describe('Search', () => {
  it('starts with guidance and no fake results', async () => {
    const { fixture } = setup(SearchComponent);
    const text = await render(fixture);
    expect(text).toContain('Search across your meetings');
    expect(text).not.toMatch(INTERNAL);
  });
});

describe('App shell', () => {
  it('has no administration or technology wording in the navigation', async () => {
    const { fixture, http } = setup(AppComponent);
    fixture.detectChanges();
    http.expectOne(`${api}/system/status`).flush({ status: 'available', can_view_meetings: true, can_upload: true, can_search: true });
    const text = await render(fixture);
    for (const label of ['Dashboard', 'Meetings', 'Upload recording', 'Search']) expect(text).toContain(label);
    expect(text).not.toMatch(/settings|admin/i);
    expect(text).not.toMatch(INTERNAL);
  });
});
