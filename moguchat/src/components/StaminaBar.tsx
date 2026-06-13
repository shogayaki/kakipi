import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { theme } from '../theme';

type Props = {
  value: number;
  max: number;
  secondsToNext: number;
  adsRemaining: number;
  onWatchAd: () => void;
};

const fmt = (sec: number): string => {
  if (sec <= 0) return '--:--';
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
};

export function StaminaBar({ value, max, secondsToNext, adsRemaining, onWatchAd }: Props) {
  const ratio = Math.max(0, Math.min(1, value / max));
  const full = value >= max;
  return (
    <View style={styles.wrap}>
      <View style={styles.row}>
        <Text style={styles.label}>⚡ スタミナ</Text>
        <Text style={styles.value}>
          {value} / {max}
        </Text>
        {!full && (
          <Text style={styles.timer}>次の回復まで {fmt(secondsToNext)}</Text>
        )}
      </View>
      <View style={styles.track}>
        <View style={[styles.fill, { width: `${ratio * 100}%` }]} />
      </View>
      <TouchableOpacity
        style={[styles.adBtn, (full || adsRemaining <= 0) && styles.adBtnDisabled]}
        onPress={onWatchAd}
        disabled={full || adsRemaining <= 0}
      >
        <Text style={styles.adBtnText}>
          {full
            ? 'スタミナは満タンです'
            : adsRemaining <= 0
            ? '本日の広告回復は上限です'
            : `📺 広告を見て回復 (残り${adsRemaining}回)`}
        </Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: {
    backgroundColor: theme.colors.surface,
    padding: theme.spacing(1.5),
    borderRadius: theme.radius,
    gap: theme.spacing(1),
  },
  row: { flexDirection: 'row', alignItems: 'center', gap: theme.spacing(1) },
  label: { color: theme.colors.text, fontWeight: '700' },
  value: { color: theme.colors.accent, fontWeight: '700' },
  timer: { color: theme.colors.textMuted, marginLeft: 'auto', fontSize: 12 },
  track: {
    height: 10,
    backgroundColor: theme.colors.surfaceAlt,
    borderRadius: 6,
    overflow: 'hidden',
  },
  fill: { height: '100%', backgroundColor: theme.colors.accent },
  adBtn: {
    backgroundColor: theme.colors.primary,
    paddingVertical: theme.spacing(1),
    borderRadius: theme.radius,
    alignItems: 'center',
  },
  adBtnDisabled: { backgroundColor: theme.colors.surfaceAlt },
  adBtnText: { color: theme.colors.text, fontWeight: '700' },
});
