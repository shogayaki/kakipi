import {
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { CHARACTERS, type Character } from '../characters';
import { theme } from '../theme';

type Props = {
  onSelect: (character: Character) => void;
};

export function CharacterSelectScreen({ onSelect }: Props) {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.title}>MoguChat</Text>
        <Text style={styles.subtitle}>今日は誰と話す？</Text>
      </View>
      <ScrollView contentContainerStyle={styles.list}>
        {CHARACTERS.map((c) => (
          <TouchableOpacity
            key={c.id}
            style={[styles.card, { borderColor: c.color }]}
            onPress={() => onSelect(c)}
            activeOpacity={0.85}
          >
            <Text style={[styles.emoji, { color: c.color }]}>{c.emoji}</Text>
            <View style={styles.cardBody}>
              <Text style={styles.name}>{c.name}</Text>
              <Text style={[styles.attribute, { color: c.color }]}>{c.attribute}</Text>
              <Text style={styles.tagline}>{c.tagline}</Text>
            </View>
          </TouchableOpacity>
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.bg },
  header: {
    paddingTop: theme.spacing(7),
    paddingHorizontal: theme.spacing(3),
    paddingBottom: theme.spacing(2),
  },
  title: { color: theme.colors.primary, fontSize: 34, fontWeight: '800' },
  subtitle: { color: theme.colors.textMuted, fontSize: 16, marginTop: 4 },
  list: { padding: theme.spacing(2), gap: theme.spacing(2) },
  card: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: theme.colors.surface,
    borderRadius: theme.radius,
    borderWidth: 2,
    padding: theme.spacing(2),
    gap: theme.spacing(2),
  },
  emoji: { fontSize: 44 },
  cardBody: { flex: 1, gap: 2 },
  name: { color: theme.colors.text, fontSize: 20, fontWeight: '700' },
  attribute: { fontSize: 13, fontWeight: '700' },
  tagline: { color: theme.colors.textMuted, fontSize: 14, marginTop: 4 },
});
