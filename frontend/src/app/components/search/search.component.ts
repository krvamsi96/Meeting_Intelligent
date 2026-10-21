import { Component, DestroyRef, OnInit, inject, signal } from '@angular/core';
import { DatePipe } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { MeetingService } from '../../services/meeting.service';
import { SearchResult } from '../../models/api.models';
import { friendlyError } from '../../core/errors';
import { formatTimestamp, speakerLabel } from '../../core/format';
import { IconComponent } from '../../shared/icon.component';

/** How closely a passage matches - a ranking aid, not a measure of correctness. */
export function relevanceLabel(similarity: number): string {
  if (similarity >= 0.55) return 'Close match';
  if (similarity >= 0.35) return 'Related';
  return 'Loosely related';
}

@Component({
  selector: 'app-search',
  imports: [FormsModule, RouterLink, DatePipe, IconComponent],
  templateUrl: './search.component.html',
  styleUrl: './search.component.css',
})
export class SearchComponent implements OnInit {
  private readonly meetings = inject(MeetingService);
  private readonly route = inject(ActivatedRoute);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  protected query = '';
  protected readonly searching = signal(false);
  protected readonly searched = signal('');
  protected readonly results = signal<SearchResult[]>([]);
  protected readonly error = signal('');

  protected readonly speakerLabel = speakerLabel;
  protected readonly formatTimestamp = formatTimestamp;
  protected readonly relevanceLabel = relevanceLabel;

  ngOnInit(): void {
    this.route.queryParamMap.pipe(takeUntilDestroyed(this.destroyRef)).subscribe((q) => {
      const text = (q.get('q') ?? '').trim();
      this.query = text;
      if (text.length >= 2) this.run(text);
    });
  }

  protected submit(): void {
    const text = this.query.trim();
    if (text.length < 2) {
      this.error.set('Type at least two characters to search.');
      return;
    }
    this.router.navigate([], { relativeTo: this.route, queryParams: { q: text } });
  }

  private run(text: string): void {
    this.searching.set(true);
    this.error.set('');
    this.meetings.search(text).subscribe({
      next: (res) => {
        this.results.set(res.results);
        this.searched.set(text);
        this.searching.set(false);
      },
      error: (err) => {
        this.results.set([]);
        this.searched.set('');
        this.error.set(friendlyError(err, 'Search is not available right now. Please try again later.'));
        this.searching.set(false);
      },
    });
  }

  /** Opens the meeting's transcript filtered to the start of the matched passage. */
  protected transcriptLink(r: SearchResult): { tab: string; q: string } {
    return { tab: 'transcript', q: r.text.split(/\s+/).slice(0, 6).join(' ') };
  }
}
