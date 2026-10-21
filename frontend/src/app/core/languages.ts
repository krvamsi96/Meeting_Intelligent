/** Language display names for the codes the API returns (it sends ISO 639-1 codes such as "en", "hi"). */
const NAMES: Record<string, string> = {
  en: 'English', hi: 'Hindi', te: 'Telugu', ta: 'Tamil', kn: 'Kannada', ml: 'Malayalam', bn: 'Bengali',
  mr: 'Marathi', gu: 'Gujarati', pa: 'Punjabi', or: 'Odia', as: 'Assamese', ur: 'Urdu', ne: 'Nepali',
  fr: 'French', es: 'Spanish', de: 'German', pt: 'Portuguese', ru: 'Russian', ja: 'Japanese', ko: 'Korean',
  zh: 'Chinese', ar: 'Arabic', it: 'Italian', nl: 'Dutch',
};

export function languageName(code: string | null | undefined): string | null {
  if (!code) return null;
  const c = code.trim().toLowerCase();
  return NAMES[c] ?? NAMES[c.split('-')[0]] ?? c.toUpperCase();
}
