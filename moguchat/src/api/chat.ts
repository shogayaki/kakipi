// AI中継サーバーを呼び出すクライアント。
// APIキーはサーバー側のみが保持し、クライアントは characterId と会話履歴のみ送る。
import { SERVER_URL } from '../config';

export type ChatRole = 'user' | 'assistant';

export type ChatMessage = {
  role: ChatRole;
  content: string;
};

export type SendResult =
  | { ok: true; reply: string }
  | { ok: false; error: string };

export async function sendChat(
  characterId: string,
  messages: ChatMessage[]
): Promise<SendResult> {
  try {
    const res = await fetch(`${SERVER_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ characterId, messages }),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      return { ok: false, error: `サーバーエラー (${res.status}) ${text}`.trim() };
    }
    const data = (await res.json()) as { reply?: string };
    if (!data.reply) return { ok: false, error: '応答が空でした' };
    return { ok: true, reply: data.reply };
  } catch (e) {
    return {
      ok: false,
      error:
        'サーバーに接続できませんでした。server/ を起動しているか、EXPO_PUBLIC_SERVER_URL を確認してください。',
    };
  }
}
