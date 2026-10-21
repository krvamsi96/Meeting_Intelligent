import { Component, OnInit, computed, inject, input, signal } from '@angular/core';
import { NgTemplateOutlet } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { MeetingService } from '../../services/meeting.service';
import { NotificationService } from '../../services/notification.service';
import { ChatSource, Decision, FinalMeeting, FinalPending } from '../../models/api.models';
import { friendlyError } from '../../core/errors';
import { deadlineLabel, formatDuration, formatTimestamp, ownerLabel, speakerLabel, toSeconds } from '../../core/format';
import { languageName } from '../../core/languages';
import { toTextBlocks } from '../../core/text-blocks';
import { IconComponent } from '../../shared/icon.component';
import { StatusBadgeComponent } from '../../shared/status-badge.component';

type Tab = 'summary' | 'transcript' | 'minutes' | 'decisions' | 'actions' | 'ask';
const TABS: Tab[] = ['summary', 'transcript', 'minutes', 'decisions', 'actions', 'ask'];

interface ChatMessage {
  role: 'user' | 'assistant' | 'error';
  text: string;
  sources?: ChatSource[];
}

@Component({
  selector: 'app-meeting-detail',
  imports: [FormsModule, NgTemplateOutlet, RouterLink, IconComponent, StatusBadgeComponent],
  templateUrl: './meeting-detail.component.html',
  styleUrl: './meeting-detail.component.css',
})
export class MeetingDetailComponent implements OnInit {
  private readonly meetings = inject(MeetingService);
  private readonly notify = inject(NotificationService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);

  readonly id = input.required<string>();

  protected readonly loading = signal(true);
  protected readonly loadError = signal('');
  protected readonly notFound = signal(false);
  protected readonly data = signal<FinalMeeting | null>(null);
  protected readonly pending = signal<FinalPending | null>(null);
  protected readonly tab = signal<Tab>('summary');
  protected readonly transcriptQuery = signal('');
  protected readonly exporting = signal<string | null>(null);
  protected readonly retrying = signal(false);

  protected readonly chat = signal<ChatMessage[]>([]);
  protected readonly asking = signal(false);
  protected question = '';
  private chatSession: string | null = null;

  protected readonly suggestions = ['What were the main points discussed?', 'What decisions were made?', 'Which tasks were assigned, and to whom?'];

  protected readonly formatTimestamp = formatTimestamp;
  protected readonly speakerLabel = speakerLabel;
  protected readonly ownerLabel = ownerLabel;
  protected readonly deadlineLabel = deadlineLabel;

  protected readonly title = computed(() => this.data()?.meeting.title?.trim() || `Meeting ${this.id()}`);
  protected readonly duration = computed(() => formatDuration(this.data()?.meeting.duration_seconds));
  protected readonly language = computed(() => {
    const m = this.data()?.meeting;
    return m?.source_language_name || languageName(m?.source_language) || null;
  });
  protected readonly speakerCount = computed(() => {
    const m = this.data()?.meeting;
    return m?.speaker_count || m?.participant_count || null;
  });
  protected readonly summaryBlocks = computed(() => toTextBlocks(this.data()?.summary));
  protected readonly minutesBlocks = computed(() => toTextBlocks(this.data()?.mom));
  protected readonly transcript = computed(() => {
    const q = this.transcriptQuery().trim().toLowerCase();
    const all = this.data()?.transcript ?? [];
    return q ? all.filter((s) => s.text.toLowerCase().includes(q) || speakerLabel(s.speaker).toLowerCase().includes(q)) : all;
  });
  protected readonly speakingTime = computed(() =>
    (this.data()?.speaker_duration ?? []).filter((s) => s.duration_seconds > 0).map((s) => ({
      label: speakerLabel(s.speaker_id),
      time: formatDuration(s.duration_seconds),
      percent: Math.round(s.percentage),
    })),
  );
  /** Agenda topics in order, each with the decisions recorded for it; decisions on no listed topic come last. */
  protected readonly agenda = computed(() => {
    const d = this.data();
    if (!d) return { topics: [], unlinked: [] as Decision[] };
    const byTopic = new Map<string, Decision[]>();
    for (const dec of d.decisions ?? []) {
      byTopic.set(dec.agenda_id, [...(byTopic.get(dec.agenda_id) ?? []), dec]);
    }
    const topics = [...(d.agenda ?? [])].sort((a, b) => a.order - b.order).map((a) => ({
      id: a.agenda_id, title: a.title, decisions: byTopic.get(a.agenda_id) ?? [],
    }));
    const known = new Set(topics.map((t) => t.id));
    return { topics, unlinked: (d.decisions ?? []).filter((x) => !known.has(x.agenda_id)) };
  });
  protected readonly decisionCount = computed(() => this.data()?.decisions?.length ?? 0);
  protected readonly tabs = computed((): { id: Tab; label: string; count: number | null }[] => {
    const d = this.data();
    return [
      { id: 'summary', label: 'Summary', count: null },
      { id: 'transcript', label: 'Transcript', count: d?.transcript.length ?? 0 },
      { id: 'minutes', label: 'Minutes', count: null },
      { id: 'decisions', label: 'Agenda & decisions', count: this.decisionCount() },
      { id: 'actions', label: 'Action items', count: d?.action_items.length ?? 0 },
      { id: 'ask', label: 'Ask a question', count: null },
    ];
  });

  ngOnInit(): void {
    const requested = this.route.snapshot.queryParamMap.get('tab') as Tab | null;
    if (requested && TABS.includes(requested)) this.tab.set(requested);
    const q = this.route.snapshot.queryParamMap.get('q');
    if (q) this.transcriptQuery.set(q);
    this.load();
  }

  protected load(): void {
    this.loading.set(true);
    this.loadError.set('');
    this.notFound.set(false);
    this.meetings.getMeeting(this.id()).subscribe({
      next: (res) => {
        if (res.kind === 'pending') {
          this.pending.set(res.pending);
          this.data.set(null);
        } else {
          this.pending.set(null);
          this.data.set(res.meeting);
        }
        this.loading.set(false);
      },
      error: (err) => {
        this.loading.set(false);
        if (err?.status === 404) this.notFound.set(true);
        else this.loadError.set(friendlyError(err, 'Your meeting results could not be loaded. Please try again.'));
      },
    });
  }

  protected selectTab(t: Tab): void {
    this.tab.set(t);
    this.router.navigate([], { relativeTo: this.route, queryParams: { tab: t === 'summary' ? null : t }, queryParamsHandling: 'merge', replaceUrl: true });
  }

  protected onTabKey(event: KeyboardEvent): void {
    if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
    const i = TABS.indexOf(this.tab());
    const next = TABS[(i + (event.key === 'ArrowRight' ? 1 : TABS.length - 1)) % TABS.length];
    this.selectTab(next);
    document.getElementById(`tab-${next}`)?.focus();
  }

  protected retry(): void {
    this.retrying.set(true);
    this.meetings.retry(this.id()).subscribe({
      next: () => {
        this.retrying.set(false);
        this.notify.success('Processing has been restarted.');
        this.router.navigate(['/meetings', this.id(), 'progress']);
      },
      error: (err) => {
        this.retrying.set(false);
        this.notify.error(friendlyError(err, 'Processing could not be restarted. Please try again.'));
      },
    });
  }

  protected download(format: 'docx' | 'xlsx'): void {
    this.exporting.set(format);
    this.meetings.download(this.id(), format).subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${this.title().replace(/[^\w\- ]+/g, '').trim().slice(0, 80) || 'meeting'}.${format}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
        this.exporting.set(null);
        this.notify.success('Your report has been downloaded.');
      },
      error: (err) => {
        this.exporting.set(null);
        this.notify.error(friendlyError(err, 'The report could not be created. Please try again.'));
      },
    });
  }

  protected printReport(): void {
    window.open(this.meetings.printableReportUrl(this.id()), '_blank', 'noopener');
  }

  protected ask(text?: string): void {
    const q = (text ?? this.question).trim();
    if (!q || this.asking()) return;
    this.question = '';
    this.chat.update((m) => [...m, { role: 'user', text: q }]);
    this.asking.set(true);
    this.meetings.ask(this.id(), q, this.chatSession).subscribe({
      next: (reply) => {
        this.chatSession = reply.session_id ?? this.chatSession;
        this.chat.update((m) => [...m, { role: 'assistant', text: reply.content, sources: (reply.sources ?? []).slice(0, 3) }]);
        this.asking.set(false);
      },
      error: (err) => {
        this.chat.update((m) => [...m, { role: 'error', text: friendlyError(err, 'This question could not be answered. Please try again.') }]);
        this.asking.set(false);
      },
    });
  }

  protected sourceSpeaker(s: ChatSource): string | null {
    const raw = s.speaker ?? s.speakers ?? null;
    return raw ? raw.split(/,\s*/).map(speakerLabel).join(', ') : null;
  }

  protected sourceTime(s: ChatSource): string | null {
    return toSeconds(s.start_time ?? null) === null ? null : formatTimestamp(s.start_time ?? null);
  }

  protected openTranscriptAt(query: string): void {
    this.transcriptQuery.set(query);
    this.selectTab('transcript');
  }
}
