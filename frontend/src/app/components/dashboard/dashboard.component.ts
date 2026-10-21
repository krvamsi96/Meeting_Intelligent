import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import { MeetingService } from '../../services/meeting.service';
import { MeetingListPage } from '../../models/api.models';
import { friendlyError } from '../../core/errors';
import { formatDuration, meetingName } from '../../core/format';
import { IconComponent } from '../../shared/icon.component';
import { StatusBadgeComponent } from '../../shared/status-badge.component';

@Component({
  selector: 'app-dashboard',
  imports: [RouterLink, DatePipe, IconComponent, StatusBadgeComponent],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.css',
})
export class DashboardComponent implements OnInit {
  private readonly meetings = inject(MeetingService);

  protected readonly loading = signal(true);
  protected readonly error = signal('');
  protected readonly page = signal<MeetingListPage | null>(null);

  /** Counts over ALL meetings (from the backend), not over the rows shown below. */
  protected readonly counts = computed(() => {
    const c = this.page()?.status_counts;
    return {
      total: c ? c.completed + c.processing + c.failed : 0,
      completed: c?.completed ?? 0,
      processing: c?.processing ?? 0,
      failed: c?.failed ?? 0,
    };
  });

  protected readonly stats = computed(() => {
    const c = this.counts();
    return [
      { label: 'All meetings', value: c.total, icon: 'meetings', tone: 'neutral', link: null },
      { label: 'Completed', value: c.completed, icon: 'checkCircle', tone: 'success', link: 'completed' },
      { label: 'Processing', value: c.processing, icon: 'clock', tone: 'info', link: 'processing' },
      { label: 'Needs attention', value: c.failed, icon: 'alert', tone: 'danger', link: 'failed' },
    ];
  });

  protected readonly formatDuration = formatDuration;
  protected readonly meetingName = meetingName;

  ngOnInit(): void {
    this.load();
  }

  protected load(): void {
    this.loading.set(true);
    this.error.set('');
    this.meetings.listMeetings({ page: 1, limit: 5, sort: 'newest' }).subscribe({
      next: (p) => {
        this.page.set(p);
        this.loading.set(false);
      },
      error: (err) => {
        this.error.set(friendlyError(err, 'Your meetings could not be loaded. Please try again.'));
        this.loading.set(false);
      },
    });
  }
}
