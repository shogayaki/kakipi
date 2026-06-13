// 実行時設定。AI中継サーバーのURLは EXPO_PUBLIC_ 環境変数で上書きできる。
// 例) EXPO_PUBLIC_SERVER_URL=http://192.168.0.10:8787 npm run web
export const SERVER_URL =
  process.env.EXPO_PUBLIC_SERVER_URL ?? 'http://localhost:8787';

// スタミナ制の設定値（企画書 docs/moguchat-mvp-plan.md と対応）
export const STAMINA = {
  MAX: 20,
  COST_PER_MESSAGE: 1,
  /** 自然回復: この間隔ごとに +1 */
  RECOVER_INTERVAL_MS: 15 * 60 * 1000,
  /** リワード広告1本あたりの回復量 */
  AD_RECOVER_AMOUNT: 10,
  /** 1日あたりの広告回復回数の上限 */
  AD_DAILY_LIMIT: 10,
} as const;
