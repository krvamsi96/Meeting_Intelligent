import { Component, DestroyRef, OnInit, inject, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MeetingService } from '../../services/meeting.service';
import { NotificationService } from '../../services/notification.service';
import { MeetingListItem, MeetingListPage, MeetingSort, MeetingState } from '../../models/api.models';
import { friendlyError } from '../../core/errors';
import { formatDuration, meetingName } from '../../core/format';
import { languageName } from '../../core/languages';
import { IconComponent } from '../../shared/icon.component';
import { StatusBadgeComponent } from '../../shared/status-badge.component';

const STATUSES: MeetingState[] = ['completed', 'processing', 'failed'];
const SORTS: MeetingSort[] = ['newest', 'oldest', 'title', 'duration'];

@Component({
  selector: 'app-meeting-list',
  imports: [FormsModule, RouterLink, DatePipe, IconComponent, StatusBadgeComponent],
  templateUrl: './meeting-list.component.html',
  styleUrl: './meeting-list.component.css',
})
export class MeetingListComponent implements OnInit {
  private readonly meetings = inject(MeetingService);
  private readonly notify = inject(NotificationService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  protected readonly loading = signal(true);
  protected readonly error = signal('');
  protected readonly page = signal<MeetingListPage | null>(null);
  protected readonly pendingDelete = signal<MeetingListItem | null>(null);
  protected readonly deleting = signal(false);
  protected readonly retrying = signal<number | null>(null);

  protected searchText = '';
  protected status: MeetingState | null = null;
  protected sort: MeetingSort = 'newest';
  protected pageNumber = 1;
  protected readonly pageSize = 10;

  protected readonly filters: { value: MeetingState | null; label: string }[] = [
    { value: null, label: 'All' },
    { value: 'completed', label: 'Completed' },
    { value: 'processing', label: 'Processing' },
    { value: 'failed', label: 'Needs attention' },
  ];
  protected readonly sorts: { value: MeetingSort; label: string }[] = [
    { value: 'newest', label: 'Newest first' },
    { value: 'oldest', label: 'Oldest first' },
    { value: 'title', label: 'Title (A–Z)' },
    { value: 'duration', label: 'Longest first' },
  ];

  protected readonly formatDuration = formatDuration;
  protected readonly meetingName = meetingName;
  protected readonly languageName = languageName;

  ngOnInit(): void {
    // The URL holds the filters, so refresh / back / dashboard links all land on the same list.
    this.route.queryParamMap.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((q) => {
      const status = q.get('status') as MeetingState | null;
      const sort = q.get('sort') as MeetingSort | null;
      this.status = status && STATUSES.includes(status) ? status : null;
      this.sort = sort && SORTS.includes(sort) ? sort : 'newest';
      this.searchText = q.get('q') ?? '';
      this.pageNumber = Math.max(1, Number(q.get('page')) || 1);
      this.load();
    });
  }

  protected load(): void {
    this.loading.set(true);
    this.error.set('');
    this.meetings
      .listMeetings({ page: this.pageNumber, limit: this.pageSize, status: this.status, search: this.searchText, sort: this.sort })
      .subscribe({
        next: (p) => {
          this.page.set(p);
          this.loading.set(false);
          if (p.items.length === 0 && p.total > 0 && this.pageNumber > p.pages) this.goTo(p.pages);
        },
        error: (err) => {
          this.error.set(friendlyError(err, 'Your meetings could not be loaded. Please try again.'));
          this.loading.set(false);
        },
      });
  }

  private navigate(changes: Record<string, string | number | null>): void {
    const current = { q: this.searchText || null, status: this.status, sort: this.sort === 'newest' ? null : this.sort, page: this.pageNumber > 1 ? this.pageNumber : null };
    this.router.navigate([], { relativeTo: this.route, queryParams: { ...current, ...changes } });
  }

  protected applySearch(): void {
    this.navigate({ q: this.searchText.trim() || null, page: null });
  }

  protected clearSearch(): void {
    this.searchText = '';
    this.applySearch();
  }

  protected setStatus(status: MeetingState | null): void {
    this.navigate({ status, page: null });
  }

  protected setSort(sort: MeetingSort): void {
    this.navigate({ sort: sort === 'newest' ? null : sort, page: null });
  }

  protected goTo(page: number): void {
    this.navigate({ page: page > 1 ? page : null });
  }

  protected resetFilters(): void {
    this.router.navigate([], { relativeTo: this.route, queryParams: {} });
  }

  protected countFor(value: MeetingState | null): number | null {
    const c = this.page()?.status_counts;
    if (!c) return null;
    return value ? c[value] : c.completed + c.processing + c.failed;
  }

  protected openLink(m: MeetingListItem): (string | number)[] {
    return m.status === 'processing' ? ['/meetings', m.id, 'progress'] : ['/meetings', m.id];
  }

  protected confirmDelete(m: MeetingListItem): void {
    this.pendingDelete.set(m);
  }

  protected delete(): void {
    const m = this.pendingDelete();
    if (!m) return;
    this.deleting.set(true);
    this.meetings.deleteMeeting(m.id).subscribe({
      next: () => {
        this.deleting.set(false);
        this.pendingDelete.set(null);
        this.notify.success(`"${meetingName(m)}" was deleted.`);
        this.load();
      },
      error: (err) => {
        this.deleting.set(false);
        this.notify.error(friendlyError(err, 'The meeting could not be deleted. Please try again.'));
      },
    });
  }

  protected retry(m: MeetingListItem): void {
    this.retrying.set(m.id);
    this.meetings.retry(m.id).subscribe({
      next: () => {
        this.retrying.set(null);
        this.notify.success('Processing has been restarted.');
        this.router.navigate(['/meetings', m.id, 'progress']);
      },
      error: (err) => {
        this.retrying.set(null);
        this.notify.error(friendlyError(err, 'Processing could not be restarted. Please try again.'));
      },
    });
  }
}
