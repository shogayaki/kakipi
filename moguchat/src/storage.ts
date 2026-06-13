// 永続化の薄い抽象レイヤー。
// - Web では localStorage を使用
// - それ以外（ネイティブ）では当面インメモリにフォールバック
// 将来 @react-native-async-storage/async-storage に差し替えられるよう、
// すべて Promise ベースの非同期APIで統一している。

type Store = {
  getItem(key: string): Promise<string | null>;
  setItem(key: string, value: string): Promise<void>;
  removeItem(key: string): Promise<void>;
};

const memory = new Map<string, string>();

const hasLocalStorage = (): boolean => {
  try {
    return typeof globalThis !== 'undefined' && !!(globalThis as any).localStorage;
  } catch {
    return false;
  }
};

export const storage: Store = {
  async getItem(key) {
    if (hasLocalStorage()) return (globalThis as any).localStorage.getItem(key);
    return memory.has(key) ? (memory.get(key) as string) : null;
  },
  async setItem(key, value) {
    if (hasLocalStorage()) {
      (globalThis as any).localStorage.setItem(key, value);
      return;
    }
    memory.set(key, value);
  },
  async removeItem(key) {
    if (hasLocalStorage()) {
      (globalThis as any).localStorage.removeItem(key);
      return;
    }
    memory.delete(key);
  },
};
