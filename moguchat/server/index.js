// MoguChat AI中継サーバー。
// Claude API を呼び出し、キャラごとの応答を返す。APIキーはこのサーバーのみが保持する。
import express from 'express';
import cors from 'cors';
import Anthropic from '@anthropic-ai/sdk';
import { buildSystemPrompt } from './characters.js';

const PORT = process.env.PORT || 8787;
// コスト重視のチャットアプリ向けに MODEL で差し替え可能（例: claude-haiku-4-5）。
const MODEL = process.env.MODEL || 'claude-opus-4-8';

const apiKey = process.env.ANTHROPIC_API_KEY;
if (!apiKey) {
  console.warn(
    '[warn] ANTHROPIC_API_KEY が未設定です。/api/chat は503を返します。.env を確認してください。'
  );
}

const client = apiKey ? new Anthropic({ apiKey }) : null;

const app = express();
app.use(cors());
app.use(express.json({ limit: '256kb' }));

app.get('/health', (_req, res) => {
  res.json({ ok: true, model: MODEL, hasKey: Boolean(apiKey) });
});

app.post('/api/chat', async (req, res) => {
  if (!client) {
    return res.status(503).json({ error: 'サーバーにAPIキーが設定されていません。' });
  }

  const { characterId, messages } = req.body ?? {};
  const system = buildSystemPrompt(characterId);
  if (!system) {
    return res.status(400).json({ error: '不明な characterId です。' });
  }
  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: 'messages が空です。' });
  }

  // クライアントから来た履歴を Claude のメッセージ形式に正規化（防御的に整形）。
  const normalized = messages
    .filter((m) => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
    .map((m) => ({ role: m.role, content: m.content }));

  if (normalized.length === 0 || normalized[0].role !== 'user') {
    return res.status(400).json({ error: '会話履歴が不正です（先頭は user である必要があります）。' });
  }

  try {
    const response = await client.messages.create({
      model: MODEL,
      max_tokens: 512,
      // 短いロールプレイ応答なので思考はオフにし、レイテンシ・コストを抑える。
      thinking: { type: 'disabled' },
      system,
      messages: normalized,
    });

    const reply = response.content
      .filter((b) => b.type === 'text')
      .map((b) => b.text)
      .join('')
      .trim();

    return res.json({ reply, model: response.model });
  } catch (err) {
    if (err instanceof Anthropic.APIError) {
      console.error(`[error] Claude API ${err.status}:`, err.message);
      const status = err.status >= 500 ? 502 : err.status;
      return res.status(status).json({ error: 'AIの応答取得に失敗しました。' });
    }
    console.error('[error] 想定外のエラー:', err);
    return res.status(500).json({ error: 'サーバー内部エラー。' });
  }
});

app.listen(PORT, () => {
  console.log(`MoguChat server listening on http://localhost:${PORT} (model: ${MODEL})`);
});
