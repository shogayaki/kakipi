// クライアント側のキャラ「表示用」データ。
// AIの人格を決めるシステムプロンプトは漏洩・改ざん防止のためサーバー側
// (server/characters.js) に置き、ここでは characterId だけを送信する。

export type Character = {
  id: string;
  name: string;
  /** 属性タイプ（夢女子向けの王道属性） */
  attribute: string;
  /** 一覧に出すキャッチコピー */
  tagline: string;
  /** チャット開始時にキャラ側から表示する最初のメッセージ */
  greeting: string;
  emoji: string;
  color: string;
};

export const CHARACTERS: Character[] = [
  {
    id: 'haru',
    name: '陽 (はる)',
    attribute: '甘々・年下系',
    tagline: '「先輩のこと、ずっと見てました」',
    greeting: '先輩、来てくれたんですね。今日は俺のこと、いっぱい構ってくれますか？',
    emoji: '🌸',
    color: '#ff6fae',
  },
  {
    id: 'rei',
    name: '澪 (れい)',
    attribute: 'クール・大人系',
    tagline: '「…別に、心配なんてしてない」',
    greeting: '……来たのか。まあ、座れ。話くらいは聞いてやる。',
    emoji: '🌙',
    color: '#6f9bff',
  },
  {
    id: 'sho',
    name: '翔 (しょう)',
    attribute: '明るい・俺様系',
    tagline: '「お前は俺が幸せにしてやる」',
    greeting: 'よお、待ってたぜ！今日も俺と話せてラッキーだったな？',
    emoji: '⚡',
    color: '#ffd166',
  },
];

export const getCharacter = (id: string): Character | undefined =>
  CHARACTERS.find((c) => c.id === id);
