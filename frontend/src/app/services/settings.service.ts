import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';

/**
 * Administrator-only AI configuration. The admin key typed by the administrator is kept for this browser tab only
 * (sessionStorage) and sent as a header; it is never part of the built app. Cloud provider keys go to the backend,
 * which stores them encrypted and only ever returns them masked.
 */

const KEY_STORAGE = 'mi-admin-key';

export type ProviderId = 'airawat' | 'ollama' | 'groq' | 'gemini';

export interface ProviderSettings {
  base_url?: string | null;
  model?: string | null;
  whisper_model?: string | null;
  api_key?: string | null;
  timeout?: number | null;
  [key: string]: unknown;
}

export interface AISettings {
  active_provider: ProviderId | string;
  stt_provider: string | null;
  effective_stt_provider: string | null;
  stt_problem?: string;
  supported_providers: { id: ProviderId; is_cloud: boolean; supports_transcription: boolean }[];
  airawat: ProviderSettings;
  ollama: ProviderSettings;
  groq: ProviderSettings;
  gemini: ProviderSettings;
}

export interface ConnectionTest {
  connected: boolean;
  latency_ms?: number;
  message?: string;
  models?: string[];
  model_available?: boolean | null;
  transcription_model_available?: boolean | null;
}

@Injectable({ providedIn: 'root' })
export class SettingsService {
  private readonly http = inject(HttpClient);
  private readonly api = environment.apiUrl;

  get adminKey(): string {
    try {
      return sessionStorage.getItem(KEY_STORAGE) ?? '';
    } catch {
      return '';
    }
  }

  setAdminKey(key: string): void {
    try {
      if (key) sessionStorage.setItem(KEY_STORAGE, key);
      else sessionStorage.removeItem(KEY_STORAGE);
    } catch {
      /* storage unavailable (private mode): the key then lives only in memory for this request */
    }
  }

  private headers(): HttpHeaders {
    const key = this.adminKey;
    return key ? new HttpHeaders({ 'X-Admin-Key': key }) : new HttpHeaders();
  }

  getSettings(): Observable<AISettings> {
    return this.http.get<AISettings>(`${this.api}/settings/ai`, { headers: this.headers() });
  }

  saveSettings(payload: Record<string, unknown>): Observable<{ status: string; message: string }> {
    return this.http.post<{ status: string; message: string }>(`${this.api}/settings/ai`, payload, { headers: this.headers() });
  }

  testConnection(provider: ProviderId, config: Record<string, unknown>): Observable<ConnectionTest> {
    return this.http.post<ConnectionTest>(`${this.api}/settings/test`, { provider, config }, { headers: this.headers() });
  }

  listModels(provider: Exclude<ProviderId, 'airawat'>): Observable<{ provider: string; models: string[] }> {
    const params = new HttpParams().set('provider', provider);
    return this.http.get<{ provider: string; models: string[] }>(`${this.api}/settings/models`, { params, headers: this.headers() });
  }
}
