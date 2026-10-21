import { Component, OnInit, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HttpErrorResponse } from '@angular/common/http';
import { AISettings, ConnectionTest, ProviderId, SettingsService } from '../../services/settings.service';
import { NotificationService } from '../../services/notification.service';
import { IconComponent } from '../../shared/icon.component';

type EditableProvider = 'ollama' | 'groq' | 'gemini';

interface ProviderForm {
  base_url: string;
  model: string;
  whisper_model: string;
  api_key: string;          // a NEW key only; empty keeps the stored one
  masked_key: string;
}

const PROVIDER_INFO: Record<ProviderId, { name: string; where: string }> = {
  airawat: { name: 'AIRAWAT', where: 'the AIRAWAT gateway configured for this deployment' },
  ollama: { name: 'Ollama (local)', where: 'the Ollama server you configure (normally on this network)' },
  groq: { name: 'Groq (cloud)', where: "Groq's cloud service" },
  gemini: { name: 'Google Gemini (cloud)', where: "Google's cloud service" },
};

function emptyForm(): ProviderForm {
  return { base_url: '', model: '', whisper_model: '', api_key: '', masked_key: '' };
}

/** Administrator page: which AI provider generates text and which transcribes, and each provider's settings. */
@Component({
  selector: 'app-settings',
  imports: [FormsModule, IconComponent],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.css',
})
export class SettingsComponent implements OnInit {
  private readonly api = inject(SettingsService);
  private readonly notify = inject(NotificationService);

  protected readonly needsKey = signal(false);
  protected readonly loading = signal(false);
  protected readonly error = signal('');
  protected readonly settings = signal<AISettings | null>(null);
  protected readonly saving = signal(false);
  protected readonly testing = signal<ProviderId | null>(null);
  protected readonly testResult = signal<{ provider: ProviderId; result: ConnectionTest } | null>(null);
  protected readonly models = signal<Partial<Record<EditableProvider, string[]>>>({});

  protected adminKeyInput = '';
  protected llmProvider: ProviderId = 'airawat';
  protected sttProvider = '';
  protected forms: Record<EditableProvider, ProviderForm> = { ollama: emptyForm(), groq: emptyForm(), gemini: emptyForm() };
  protected readonly info = PROVIDER_INFO;
  protected readonly editable: EditableProvider[] = ['ollama', 'groq', 'gemini'];

  ngOnInit(): void {
    if (this.api.adminKey) this.load();
    else this.needsKey.set(true);
  }

  protected unlock(): void {
    this.api.setAdminKey(this.adminKeyInput.trim());
    this.adminKeyInput = '';
    this.load();
  }

  protected lock(): void {
    this.api.setAdminKey('');
    this.settings.set(null);
    this.needsKey.set(true);
  }

  private handleError(err: unknown, fallback: string): void {
    if (err instanceof HttpErrorResponse && err.status === 401) {
      this.api.setAdminKey('');
      this.needsKey.set(true);
      this.error.set('The administrator key was not accepted.');
      return;
    }
    const detail = err instanceof HttpErrorResponse ? (err.error?.message ?? err.error?.detail) : null;
    this.error.set(typeof detail === 'string' ? detail : fallback);
  }

  protected load(): void {
    this.loading.set(true);
    this.error.set('');
    this.api.getSettings().subscribe({
      next: (s) => {
        this.needsKey.set(false);
        this.settings.set(s);
        this.llmProvider = (s.active_provider as ProviderId) || 'airawat';
        this.sttProvider = s.stt_provider ?? '';
        for (const p of this.editable) {
          const cfg = s[p] ?? {};
          this.forms[p] = {
            base_url: String(cfg.base_url ?? ''),
            model: String(cfg.model ?? ''),
            whisper_model: String(cfg.whisper_model ?? ''),
            api_key: '',
            masked_key: String(cfg.api_key ?? ''),
          };
        }
        this.loading.set(false);
      },
      error: (err) => {
        this.loading.set(false);
        this.handleError(err, 'The settings could not be loaded.');
      },
    });
  }

  protected isCloud(p: ProviderId | string): boolean {
    return p === 'groq' || p === 'gemini' || p === 'airawat';
  }

  protected effectiveStt(): string | null {
    if (this.sttProvider) return this.sttProvider;
    return this.llmProvider === 'groq' || this.llmProvider === 'airawat' ? this.llmProvider : null;
  }

  /** Where meeting content goes with the current (unsaved) choice - shown before saving. */
  protected destinations(): string[] {
    const out = new Set<string>([PROVIDER_INFO[this.llmProvider].where]);
    const stt = this.effectiveStt() as ProviderId | null;
    if (stt) out.add(PROVIDER_INFO[stt].where);
    return [...out];
  }

  protected loadModels(p: EditableProvider): void {
    this.api.listModels(p).subscribe({
      next: (r) => {
        this.models.update((m) => ({ ...m, [p]: r.models }));
        this.notify.info(`${r.models.length} models are available.`);
      },
      error: (err) => this.handleError(err, 'The model list could not be read.'),
    });
  }

  private formPayload(p: EditableProvider): Record<string, unknown> {
    const f = this.forms[p];
    const out: Record<string, unknown> = {};
    if (f.base_url.trim()) out['base_url'] = f.base_url.trim();
    if (f.model.trim()) out['model'] = f.model.trim();
    if (p === 'groq' && f.whisper_model.trim()) out['whisper_model'] = f.whisper_model.trim();
    if (p !== 'ollama' && f.api_key.trim()) out['api_key'] = f.api_key.trim();
    return out;
  }

  protected test(p: ProviderId): void {
    this.testing.set(p);
    this.testResult.set(null);
    this.api.testConnection(p, p === 'airawat' ? {} : this.formPayload(p)).subscribe({
      next: (result) => {
        this.testing.set(null);
        this.testResult.set({ provider: p, result });
      },
      error: (err) => {
        this.testing.set(null);
        this.handleError(err, 'The connection test could not be run.');
      },
    });
  }

  protected save(): void {
    this.saving.set(true);
    this.error.set('');
    const payload: Record<string, unknown> = { active_provider: this.llmProvider, stt_provider: this.sttProvider };
    for (const p of this.editable) payload[p] = this.formPayload(p);
    this.api.saveSettings(payload).subscribe({
      next: () => {
        this.saving.set(false);
        this.notify.success('Settings saved. New meetings use them right away.');
        this.load();
      },
      error: (err) => {
        this.saving.set(false);
        this.handleError(err, 'The settings could not be saved.');
      },
    });
  }
}
