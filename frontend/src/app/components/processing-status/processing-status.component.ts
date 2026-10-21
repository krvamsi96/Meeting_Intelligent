import { Component, OnDestroy, OnInit, inject, input, signal } from '@angular/core';
import { Router, RouterLink } from '@angular/router';
import { MeetingService } from '../../services/meeting.service';
import { NotificationService } from '../../services/notification.service';
import { ProcessingStatus } from '../../models/api.models';
import { friendlyError } from '../../core/errors';
import { STEP_LABELS } from '../../core/status';
import { IconComponent } from '../../shared/icon.component';
import { StatusBadgeComponent } from '../../shared/status-badge.component';

const POLL_MS = 3000;
const MAX_BACKOFF_MS = 30000;

@Component({
  selector: 'app-processing-status',
  imports: [RouterLink, IconComponent, StatusBadgeComponent],
  templateUrl: './processing-status.component.html',
  styleUrl: './processing-status.component.css',
})
export class ProcessingStatusComponent implements OnInit, OnDestroy {
  private readonly meetings = inject(MeetingService);
  private readonly notify = inject(NotificationService);
  private readonly router = inject(Router);

  /** Route parameter (component input binding). */
  readonly id = input.required<string>();

  protected readonly status = signal<ProcessingStatus | null>(null);
  protected readonly loadError = signal('');
  protected readonly connectionTrouble = signal(false);
  protected readonly retrying = signal(false);
  protected readonly stepLabels = STEP_LABELS;

  private timer: ReturnType<typeof setTimeout> | null = null;
  private failures = 0;
  private destroyed = false;

  ngOnInit(): void {
    this.poll();
  }

  ngOnDestroy(): void {
    this.destroyed = true;
    this.stop();
  }

  private stop(): void {
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  private schedule(ms: number): void {
    this.stop();
    if (!this.destroyed) this.timer = setTimeout(() => this.poll(), ms);
  }

  private poll(): void {
    this.meetings.getStatus(this.id()).subscribe({
      next: (s) => {
        this.failures = 0;
        this.connectionTrouble.set(false);
        this.loadError.set('');
        this.status.set(s);
        if (s.status === 'processing') this.schedule(POLL_MS);
      },
      error: (err) => {
        if (err?.status === 404) {
          this.loadError.set('This meeting could not be found. It may have been deleted.');
          return;
        }
        this.failures++;
        if (!this.status()) {
          this.loadError.set(friendlyError(err, 'The processing status could not be loaded. Please try again.'));
        }
        this.connectionTrouble.set(true);
        this.schedule(Math.min(MAX_BACKOFF_MS, POLL_MS * 2 ** this.failures));
      },
    });
  }

  protected reload(): void {
    this.loadError.set('');
    this.failures = 0;
    this.poll();
  }

  protected retry(): void {
    this.retrying.set(true);
    this.meetings.retry(this.id()).subscribe({
      next: () => {
        this.retrying.set(false);
        this.notify.success('Processing has been restarted.');
        this.status.set(null);
        this.poll();
      },
      error: (err) => {
        this.retrying.set(false);
        this.notify.error(friendlyError(err, 'Processing could not be restarted. Please try again.'));
      },
    });
  }

  protected openResults(): void {
    this.router.navigate(['/meetings', this.id()]);
  }
}
