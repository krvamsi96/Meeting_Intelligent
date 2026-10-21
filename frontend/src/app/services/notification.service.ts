import { Injectable, signal } from '@angular/core';

export interface Toast {
  id: number;
  kind: 'success' | 'error' | 'info';
  text: string;
}

/** Short confirmation / failure messages shown in the corner of the screen. */
@Injectable({ providedIn: 'root' })
export class NotificationService {
  readonly toasts = signal<Toast[]>([]);
  private nextId = 1;

  success(text: string): void { this.push('success', text); }
  error(text: string): void { this.push('error', text, 8000); }
  info(text: string): void { this.push('info', text); }

  dismiss(id: number): void {
    this.toasts.update((list) => list.filter((t) => t.id !== id));
  }

  private push(kind: Toast['kind'], text: string, ttl = 5000): void {
    const id = this.nextId++;
    this.toasts.update((list) => [...list.slice(-3), { id, kind, text }]);
    setTimeout(() => this.dismiss(id), ttl);
  }
}
