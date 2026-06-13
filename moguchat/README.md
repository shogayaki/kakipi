# MoguChat（ひな形 / scaffold）

「無課金夢女子の救世主アプリ」MoguChat の MVP ひな形です。
企画は [`../docs/moguchat-mvp-plan.md`](../docs/moguchat-mvp-plan.md) を参照。

このひな形に含まれるもの（**UI ＋ スタミナ制 ＋ AI連携の土台**）:

- **キャラ選択画面**（公式3キャラ）／**チャット画面**／**スタミナ表示**
- **スタミナ制ロジック**: 消費・時間による自然回復・リワード広告での回復（モック）、永続化
- **AI連携の土台**: Claude API を呼ぶ中継サーバー（**APIキーはサーバー側のみ保持**）

```
moguchat/
├── App.tsx                 軽量ナビゲーション（キャラ選択 ⇄ チャット）
├── src/
│   ├── characters.ts       キャラ表示データ（人格プロンプトはサーバー側）
│   ├── config.ts           サーバーURL・スタミナ設定値
│   ├── stamina.ts          スタミナのロジック + useStamina フック
│   ├── storage.ts          永続化の抽象（web=localStorage / それ以外=メモリ）
│   ├── api/chat.ts         中継サーバー呼び出しクライアント
│   ├── components/         StaminaBar / MessageBubble
│   └── screens/            CharacterSelectScreen / ChatScreen
└── server/                 AI中継サーバー（Express + @anthropic-ai/sdk）
    ├── index.js
    ├── characters.js       キャラのシステムプロンプト（サーバー側のみ）
    └── .env.example
```

## 動かし方

### 1. AI中継サーバー

```bash
cd server
npm install
cp .env.example .env       # .env に ANTHROPIC_API_KEY を設定
npm start                  # http://localhost:8787
```

- `MODEL` 環境変数でモデルを切替できます（デフォルト `claude-opus-4-8`）。
  無課金ユーザー中心でコストを抑えたい場合は `claude-haiku-4-5` などを推奨。
- APIキー未設定でもサーバーは起動し、`/api/chat` は 503 を返します（UIの確認は可能）。

### 2. アプリ（Expo）

> このリポジトリの実行環境には Flutter は無いため、Expo（React Native）で構成しています。
> Web で最も手軽に動作確認できます。

```bash
npm install
npm run web                # ブラウザで確認（localStorage でスタミナが永続化）
# npm run ios / npm run android で実機・シミュレータ（要 Expo 環境）
```

サーバーURLを変えたい場合（実機からPCのサーバーに繋ぐ等）:

```bash
EXPO_PUBLIC_SERVER_URL=http://192.168.0.10:8787 npm run web
```

## 設計上のポイント

- **APIキーとキャラの人格プロンプトはサーバー側のみ**に置き、クライアントには出さない。
- **スタミナの消費・広告回復は本番ではサーバー側で検証**すること（現状はMVPの土台としてクライアント実装）。
- 会話履歴は直近Nターンに絞って送信し、トークンを節約。
- ネイティブで永続化が必要になったら `src/storage.ts` を
  `@react-native-async-storage/async-storage` に差し替え可能（Promise APIで統一済み）。

## 次のステップ（企画書のステップ3）

- 画像生成（立ち絵・表情差分・シチュ画像）
- シチュエーション選択
- 課金プラン（スタミナ上限解放・広告非表示）
