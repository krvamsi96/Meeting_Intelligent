import { Component, OnDestroy, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router, RouterLink } from '@angular/router';
import { HttpEventType } from '@angular/common/http';
import { Subscription } from 'rxjs';
import { MeetingService } from '../../services/meeting.service';
import { NotificationService } from '../../services/notification.service';
import { UploadOptions } from '../../models/api.models';
import { friendlyError } from '../../core/errors';
import { formatFileSize } from '../../core/format';
import { IconComponent } from '../../shared/icon.component';

/** The language the form starts with: the configured default, else '' (detected automatically, or must be chosen). */
export function initialLanguage(o: UploadOptions): string {
  return o.default_language ?? '';
}

/** Why a file cannot be uploaded (null when it can). Same rules as the server, checked before sending. */
export function validateFile(file: File, o: UploadOptions): string | null {
  const ext = ('.' + (file.name.split('.').pop() ?? '')).toLowerCase();
  if (!file.name.includes('.') || !o.accepted_extensions.includes(ext)) {
    return `This file type is not supported. Choose an audio or video recording (${o.accepted_extensions.join(', ')}).`;
  }
  if (file.size === 0) return 'This file is empty. Choose another recording.';
  if (file.size > o.max_file_size_mb * 1024 * 1024) {
    return `This file is ${formatFileSize(file.size)}; the largest file you can upload is ${o.max_file_size_mb} MB.`;
  }
  return null;
}

@Component({
  selector: 'app-upload',
  imports: [FormsModule, RouterLink, IconComponent],
  templateUrl: './upload.component.html',
  styleUrl: './upload.component.css',
})
export class UploadComponent implements OnInit, OnDestroy {
  private readonly meetings = inject(MeetingService);
  private readonly notify = inject(NotificationService);
  private readonly router = inject(Router);

  protected readonly options = signal<UploadOptions | null>(null);
  protected readonly optionsError = signal('');
  protected readonly file = signal<File | null>(null);
  protected readonly fileError = signal('');
  protected readonly dragging = signal(false);
  protected readonly uploading = signal(false);
  protected readonly progress = signal(0);
  protected readonly uploadError = signal('');

  protected title = '';
  protected language = '';
  private sub?: Subscription;

  protected readonly formatFileSize = formatFileSize;

  ngOnInit(): void {
    this.loadOptions();
  }

  ngOnDestroy(): void {
    this.sub?.unsubscribe();
  }

  protected loadOptions(): void {
    this.optionsError.set('');
    this.meetings.getUploadOptions().subscribe({
      next: (o) => {
        this.options.set(o);
        this.language = initialLanguage(o);
      },
      error: (err) => this.optionsError.set(friendlyError(err, 'The upload form could not be loaded. Please try again.')),
    });
  }

  protected accept(): string {
    return (this.options()?.accepted_extensions ?? []).join(',');
  }

  protected onDragOver(e: DragEvent): void {
    e.preventDefault();
    if (!this.uploading()) this.dragging.set(true);
  }

  protected onDragLeave(e: DragEvent): void {
    e.preventDefault();
    this.dragging.set(false);
  }

  protected onDrop(e: DragEvent): void {
    e.preventDefault();
    this.dragging.set(false);
    const f = e.dataTransfer?.files?.[0];
    if (f && !this.uploading()) this.choose(f);
  }

  protected onPick(e: Event): void {
    const input = e.target as HTMLInputElement;
    const f = input.files?.[0];
    if (f) this.choose(f);
    input.value = '';      // choosing the same file again still fires a change
  }

  private choose(f: File): void {
    const o = this.options();
    this.uploadError.set('');
    const problem = o ? validateFile(f, o) : null;
    this.fileError.set(problem ?? '');
    this.file.set(problem ? null : f);
    if (!problem && !this.title.trim()) {
      this.title = f.name.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ').trim().slice(0, 200);
    }
  }

  protected removeFile(): void {
    this.file.set(null);
    this.fileError.set('');
    this.uploadError.set('');
  }

  protected canSubmit(): boolean {
    const o = this.options();
    return !!this.file() && !this.uploading() && !!o && (!o.language_required || !!this.language);
  }

  protected submit(): void {
    const f = this.file();
    if (!f || !this.canSubmit()) return;
    this.uploading.set(true);
    this.progress.set(0);
    this.uploadError.set('');
    this.sub = this.meetings.upload(f, this.title, this.language || null).subscribe({
      next: (event) => {
        if (event.type === HttpEventType.UploadProgress && event.total) {
          this.progress.set(Math.min(99, Math.round((100 * event.loaded) / event.total)));
        } else if (event.type === HttpEventType.Response) {
          this.uploading.set(false);
          this.progress.set(100);
          const id = event.body?.id;
          this.notify.success('Your recording was uploaded. Processing has started.');
          if (event.body?.potential_duplicate) {
            this.notify.info('A recording with the same name and size was uploaded before. Both are kept.');
          }
          this.router.navigate(id ? ['/meetings', id, 'progress'] : ['/meetings']);
        }
      },
      error: (err) => {
        this.uploading.set(false);
        this.progress.set(0);
        this.uploadError.set(friendlyError(err, 'The recording could not be uploaded. Please try again.'));
      },
    });
  }

  protected cancel(): void {
    this.sub?.unsubscribe();
    this.uploading.set(false);
    this.progress.set(0);
    this.notify.info('Upload cancelled.');
  }
}
