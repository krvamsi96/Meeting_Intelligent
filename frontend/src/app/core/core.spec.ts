import { HttpErrorResponse } from '@angular/common/http';
import { deadlineLabel, formatDuration, formatTimestamp, meetingName, ownerLabel, speakerLabel, toSeconds } from './format';
import { friendlyError, isUserSafeText } from './errors';
import { normalizeStatus, statusLabel } from './status';
import { toTextBlocks } from './text-blocks';

describe('format', () => {
  it('reads seconds from numbers, numeric strings and clock strings', () => {
    expect(toSeconds(61.624)).toBe(61.624);
    expect(toSeconds('61.624')).toBe(61.624);
    expect(toSeconds('01:02:03')).toBe(3723);
    expect(toSeconds('04:07')).toBe(247);
    expect(toSeconds('')).toBeNull();
    expect(toSeconds('abc')).toBeNull();
    expect(toSeconds(null)).toBeNull();
  });

  it('never invents a duration or timestamp', () => {
    expect(formatDuration(null)).toBe('—');
    expect(formatDuration(0)).toBe('—');
    expect(formatDuration(45)).toBe('45s');
    expect(formatDuration(725)).toBe('12m 05s');
    expect(formatDuration(3720)).toBe('1h 02m');
    expect(formatTimestamp(null)).toBe('—');
    expect(formatTimestamp('3723.4')).toBe('1:02:03');
    expect(formatTimestamp(65)).toBe('01:05');
  });

  it('labels machine speaker ids for people', () => {
    expect(speakerLabel('Speaker_1')).toBe('Speaker 1');
    expect(speakerLabel('SPEAKER_00')).toBe('Speaker 1');
    expect(speakerLabel('SPEAKER_01')).toBe('Speaker 2');
    expect(speakerLabel('Speaker_10')).toBe('Speaker 10');
    expect(speakerLabel('Priya Sharma')).toBe('Priya Sharma');
    expect(speakerLabel(null)).toBe('Unidentified speaker');
  });

  it('treats placeholder owners and deadlines as absent', () => {
    expect(ownerLabel('Unassigned')).toBeNull();
    expect(ownerLabel('Ravi')).toBe('Ravi');
    expect(ownerLabel('Speaker_1')).toBe('Speaker 1');
    expect(deadlineLabel('not_specified', null)).toBeNull();
    expect(deadlineLabel('Friday', null)).toBe('Friday');
    expect(deadlineLabel('Friday', '2026-10-24')).toBe('2026-10-24');
  });

  it('names a meeting by title, then file, then number', () => {
    expect(meetingName({ title: 'Budget review', filename: 'a.mp4', id: 3 })).toBe('Budget review');
    expect(meetingName({ title: null, filename: 'a.mp4', id: 3 })).toBe('a.mp4');
    expect(meetingName({ title: null, filename: null, id: 3 })).toBe('Meeting 3');
  });
});

describe('status', () => {
  it('maps every backend lifecycle value to one of three user states', () => {
    expect(normalizeStatus('completed')).toBe('completed');
    expect(normalizeStatus('error')).toBe('failed');
    expect(normalizeStatus('uploaded')).toBe('processing');
    expect(normalizeStatus(undefined)).toBe('processing');
    expect(statusLabel('error')).toBe('Needs attention');
    expect(statusLabel('processing')).toBe('Processing meeting');
  });

  it('never reports an unknown state as completed', () => {
    for (const s of ['', 'pending', 'weird', null]) expect(normalizeStatus(s)).not.toBe('completed');
  });
});

describe('friendlyError', () => {
  const err = (status: number, body: unknown) => new HttpErrorResponse({ status, error: body });

  it('shows a user-written server message for validation problems', () => {
    expect(friendlyError(err(400, { message: 'The file is empty.' }), 'x')).toBe('The file is empty.');
    expect(friendlyError(err(409, { detail: 'This meeting is still being processed.' }), 'x')).toBe('This meeting is still being processed.');
  });

  it('never shows technical server text', () => {
    const technical = [
      'psycopg2.errors.InFailedSqlTransaction: current transaction is aborted',
      'Groq Whisper transcription failed (HTTP 401)',
      'FileNotFoundError: /home/app/uploads/x.mp4',
      'Traceback (most recent call last)',
      'Connection to https://api.example.com failed',
    ];
    for (const t of technical) {
      expect(friendlyError(err(400, { message: t }), 'Something went wrong.')).toBe('Something went wrong.');
      expect(isUserSafeText(t)).toBe(false);
    }
  });

  it('uses the fallback for server errors and a clear message when offline', () => {
    expect(friendlyError(err(500, { message: 'An internal server error occurred.' }), 'Please try again.')).toBe('Please try again.');
    expect(friendlyError(err(0, null), 'x')).toMatch(/could not be reached/);
    expect(friendlyError(err(401, null), 'x')).toMatch(/permission/);
  });
});

describe('toTextBlocks', () => {
  it('turns light markdown into safe structured blocks', () => {
    const blocks = toTextBlocks('## Attendees\n- **Ravi**\n- Priya\n\nThe team agreed to ship.\n\n**Next steps:**\n1. Test\n2. Release');
    expect(blocks).toEqual([
      { type: 'heading', text: 'Attendees' },
      { type: 'list', items: ['Ravi', 'Priya'] },
      { type: 'paragraph', text: 'The team agreed to ship.' },
      { type: 'heading', text: 'Next steps:' },
      { type: 'list', items: ['Test', 'Release'] },
    ]);
  });

  it('keeps markup as plain text', () => {
    const blocks = toTextBlocks('<img src=x onerror=alert(1)>');
    expect(blocks).toEqual([{ type: 'paragraph', text: '<img src=x onerror=alert(1)>' }]);
  });

  it('returns nothing for empty input', () => {
    expect(toTextBlocks(null)).toEqual([]);
    expect(toTextBlocks('   \n  ')).toEqual([]);
  });
});
