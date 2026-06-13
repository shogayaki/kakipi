// スタミナ制のロジックと React フック。
// 注意: 本番ではスタミナの消費・広告回復の検証は必ずサーバー側で行うこと
// (クライアント改ざん対策)。ここはMVPの土台としてクライアント実装を提供する。
import { useCallback, useEffect, useRef, useState } from 'react';
import { STAMINA } from './config';
import { storage } from './storage';

const STORAGE_KEY = 'moguchat.stamina.v1';

export type StaminaState = {
  value: number;
  /** value を最後に確定した時刻(ms) */
  lastUpdated: number;
  /** 当日に広告で回復した回数 */
  adsWatched: number;
  /** adsWatched が属する日付 (YYYY-MM-DD) */
  adDay: string;
};

const todayKey = (now: number): string =>
  new Date(now).toISOString().slice(0, 10);

const initialState = (now: number): StaminaState => ({
  value: STAMINA.MAX,
  lastUpdated: now,
  adsWatched: 0,
  adDay: todayKey(now),
});

/** 経過時間に応じた自然回復を反映した状態を返す(純粋関数) */
export function applyNaturalRecovery(
  state: StaminaState,
  now: number
): StaminaState {
  // 日付が変わっていれば広告回数をリセット
  const day = todayKey(now);
  const adsWatched = day === state.adDay ? state.adsWatched : 0;

  if (state.value >= STAMINA.MAX) {
    return { ...state, value: STAMINA.MAX, lastUpdated: now, adsWatched, adDay: day };
  }
  const elapsed = Math.max(0, now - state.lastUpdated);
  const recovered = Math.floor(elapsed / STAMINA.RECOVER_INTERVAL_MS);
  if (recovered <= 0) {
    return { ...state, adsWatched, adDay: day };
  }
  const value = Math.min(STAMINA.MAX, state.value + recovered);
  // 端数の時間は次回に持ち越す。満タンになったら now にリセット。
  const lastUpdated =
    value >= STAMINA.MAX
      ? now
      : state.lastUpdated + recovered * STAMINA.RECOVER_INTERVAL_MS;
  return { value, lastUpdated, adsWatched, adDay: day };
}

/** 次の自然回復までの残り秒数(満タン時は0) */
export function secondsUntilNextRecovery(
  state: StaminaState,
  now: number
): number {
  if (state.value >= STAMINA.MAX) return 0;
  const elapsedInCurrent =
    (now - state.lastUpdated) % STAMINA.RECOVER_INTERVAL_MS;
  return Math.ceil((STAMINA.RECOVER_INTERVAL_MS - elapsedInCurrent) / 1000);
}

export type UseStamina = {
  value: number;
  max: number;
  secondsToNext: number;
  adsRemaining: number;
  canChat: boolean;
  ready: boolean;
  /** メッセージ送信時に消費。成功時 true */
  consume: () => boolean;
  /** リワード広告視聴で回復(モック)。成功時 true */
  watchAd: () => boolean;
};

export function useStamina(): UseStamina {
  const [state, setState] = useState<StaminaState>(() => initialState(Date.now()));
  const [ready, setReady] = useState(false);
  const stateRef = useRef(state);
  stateRef.current = state;

  const persist = useCallback((next: StaminaState) => {
    stateRef.current = next;
    setState(next);
    void storage.setItem(STORAGE_KEY, JSON.stringify(next));
  }, []);

  // 初回ロード
  useEffect(() => {
    let mounted = true;
    (async () => {
      const raw = await storage.getItem(STORAGE_KEY);
      const now = Date.now();
      let loaded = initialState(now);
      if (raw) {
        try {
          loaded = applyNaturalRecovery(JSON.parse(raw) as StaminaState, now);
        } catch {
          // 壊れたデータは初期化
        }
      }
      if (mounted) {
        persist(loaded);
        setReady(true);
      }
    })();
    return () => {
      mounted = false;
    };
  }, [persist]);

  // 1秒ごとに自然回復・カウントダウンを再計算
  const [, forceTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => {
      const next = applyNaturalRecovery(stateRef.current, Date.now());
      if (next.value !== stateRef.current.value || next.adsWatched !== stateRef.current.adsWatched) {
        persist(next);
      }
      forceTick((t) => (t + 1) % 1_000_000);
    }, 1000);
    return () => clearInterval(id);
  }, [persist]);

  const now = Date.now();
  const current = applyNaturalRecovery(state, now);

  const consume = useCallback((): boolean => {
    const s = applyNaturalRecovery(stateRef.current, Date.now());
    if (s.value < STAMINA.COST_PER_MESSAGE) {
      persist(s);
      return false;
    }
    persist({
      ...s,
      value: s.value - STAMINA.COST_PER_MESSAGE,
      // 満タンから減った瞬間に回復タイマーを開始
      lastUpdated:
        s.value >= STAMINA.MAX ? Date.now() : s.lastUpdated,
    });
    return true;
  }, [persist]);

  const watchAd = useCallback((): boolean => {
    const s = applyNaturalRecovery(stateRef.current, Date.now());
    if (s.adsWatched >= STAMINA.AD_DAILY_LIMIT) return false;
    persist({
      ...s,
      value: Math.min(STAMINA.MAX, s.value + STAMINA.AD_RECOVER_AMOUNT),
      adsWatched: s.adsWatched + 1,
    });
    return true;
  }, [persist]);

  return {
    value: current.value,
    max: STAMINA.MAX,
    secondsToNext: secondsUntilNextRecovery(current, now),
    adsRemaining: Math.max(0, STAMINA.AD_DAILY_LIMIT - current.adsWatched),
    canChat: current.value >= STAMINA.COST_PER_MESSAGE,
    ready,
    consume,
    watchAd,
  };
}
