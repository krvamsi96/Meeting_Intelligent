/**
 * Minutes of meeting arrive as plain text with light Markdown (headings, bullets, **bold**). They are split into
 * structured blocks rendered with normal Angular bindings - never inserted as HTML - so generated text cannot
 * inject markup.
 */

export type TextBlock =
  | { type: 'heading'; text: string }
  | { type: 'paragraph'; text: string }
  | { type: 'list'; items: string[] };

const HEADING = /^\s{0,3}#{1,6}\s+(.*)$/;
const BULLET = /^\s*(?:[-*•]|\d+[.)])\s+(.*)$/;
const BOLD_LINE = /^\s*\*\*(.+?)\*\*:?\s*$/;

/** Remove inline Markdown emphasis/code markers, keeping the words. */
export function plainInline(text: string): string {
  return text.replace(/\*\*(.+?)\*\*/g, '$1').replace(/__(.+?)__/g, '$1').replace(/`([^`]+)`/g, '$1')
    .replace(/(^|\s)\*(\S[^*]*?)\*(?=\s|$)/g, '$1$2').replace(/\s+/g, ' ').trim();
}

export function toTextBlocks(source: string | null | undefined): TextBlock[] {
  const blocks: TextBlock[] = [];
  let paragraph: string[] = [];
  let list: string[] | null = null;

  const flushParagraph = () => {
    if (paragraph.length) blocks.push({ type: 'paragraph', text: plainInline(paragraph.join(' ')) });
    paragraph = [];
  };
  const flushList = () => {
    if (list && list.length) blocks.push({ type: 'list', items: list });
    list = null;
  };

  for (const raw of (source ?? '').split(/\r?\n/)) {
    const line = raw.trimEnd();
    if (!line.trim() || /^\s*([-*_])\1{2,}\s*$/.test(line)) {
      flushParagraph();
      flushList();
      continue;
    }
    const heading = HEADING.exec(line) || BOLD_LINE.exec(line);
    if (heading) {
      flushParagraph();
      flushList();
      blocks.push({ type: 'heading', text: plainInline(heading[1]) });
      continue;
    }
    const bullet = BULLET.exec(line);
    if (bullet) {
      flushParagraph();
      (list ??= []).push(plainInline(bullet[1].replace(/^\[[ xX]\]\s*/, '')));
      continue;
    }
    flushList();
    paragraph.push(line.trim());
  }
  flushParagraph();
  flushList();
  return blocks;
}
