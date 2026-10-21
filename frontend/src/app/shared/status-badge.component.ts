import { ChangeDetectionStrategy, Component, computed, input } from '@angular/core';
import { statusBadgeClass, statusLabel } from '../core/status';

@Component({
  selector: 'app-status-badge',
  changeDetection: ChangeDetectionStrategy.OnPush,
  template: `<span class="badge" [class]="'badge ' + cls()">{{ label() }}</span>`,
})
export class StatusBadgeComponent {
  readonly status = input<string | null | undefined>(null);
  protected readonly label = computed(() => statusLabel(this.status()));
  protected readonly cls = computed(() => statusBadgeClass(this.status()));
}
