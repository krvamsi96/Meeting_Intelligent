import { Component, DestroyRef, OnInit, inject, signal } from '@angular/core';
import { NavigationEnd, Router, RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';
import { takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { filter } from 'rxjs';
import { MeetingService } from './services/meeting.service';
import { NotificationService } from './services/notification.service';
import { ServiceStatus } from './models/api.models';
import { IconComponent } from './shared/icon.component';

const THEME_KEY = 'mi-theme';
const RECHECK_MS = 15000;

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive, IconComponent],
  templateUrl: './app.component.html',
  styleUrl: './app.component.css',
})
export class AppComponent implements OnInit {
  private readonly meetings = inject(MeetingService);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);
  protected readonly notifications = inject(NotificationService);

  protected readonly menuOpen = signal(false);
  protected readonly dark = signal(false);
  /** null until known; only a problem is ever shown. */
  protected readonly service = signal<ServiceStatus | null>(null);
  protected readonly serviceUnreachable = signal(false);
  private recheck: ReturnType<typeof setTimeout> | null = null;

  protected readonly nav = [
    { path: '/dashboard', label: 'Dashboard', icon: 'dashboard' },
    { path: '/meetings', label: 'Meetings', icon: 'meetings' },
    { path: '/upload', label: 'Upload recording', icon: 'upload' },
    { path: '/search', label: 'Search', icon: 'search' },
  ];

  ngOnInit(): void {
    this.applyTheme(this.readTheme());
    this.checkService();
    this.router.events
      .pipe(filter((e) => e instanceof NavigationEnd), takeUntilDestroyed(this.destroyRef))
      .subscribe(() => this.menuOpen.set(false));
    this.destroyRef.onDestroy(() => this.recheck && clearTimeout(this.recheck));
  }

  protected checkService(): void {
    if (this.recheck) clearTimeout(this.recheck);
    this.recheck = null;
    this.meetings.serviceStatus().subscribe({
      next: (s) => {
        this.service.set(s);
        this.serviceUnreachable.set(false);
      },
      error: () => {
        // Keep checking, so the banner clears by itself once the service is back.
        this.serviceUnreachable.set(true);
        this.recheck = setTimeout(() => this.checkService(), RECHECK_MS);
      },
    });
  }

  protected toggleTheme(): void {
    this.applyTheme(!this.dark());
    try {
      localStorage.setItem(THEME_KEY, this.dark() ? 'dark' : 'light');
    } catch {
      /* preference simply not remembered */
    }
  }

  private readTheme(): boolean {
    try {
      const saved = localStorage.getItem(THEME_KEY);
      if (saved) return saved === 'dark';
    } catch {
      /* fall through to the system preference */
    }
    return typeof matchMedia === 'function' && matchMedia('(prefers-color-scheme: dark)').matches;
  }

  private applyTheme(dark: boolean): void {
    this.dark.set(dark);
    document.documentElement.setAttribute('data-theme', dark ? 'dark' : 'light');
  }
}
