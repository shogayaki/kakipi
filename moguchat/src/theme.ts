// アプリ全体の配色・余白などの共通テーマ
export const theme = {
  colors: {
    bg: '#1b1130',
    surface: '#2a1d4a',
    surfaceAlt: '#36285c',
    primary: '#ff6fae',
    primaryDark: '#d94e8c',
    accent: '#ffd166',
    text: '#ffffff',
    textMuted: '#c9bfe0',
    bubbleUser: '#ff6fae',
    bubbleChar: '#36285c',
    danger: '#ff5d6c',
  },
  spacing: (n: number) => n * 8,
  radius: 16,
} as const;

export type Theme = typeof theme;
