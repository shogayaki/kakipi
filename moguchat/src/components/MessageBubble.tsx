import { StyleSheet, Text, View } from 'react-native';
import { theme } from '../theme';

type Props = {
  role: 'user' | 'assistant';
  content: string;
  pending?: boolean;
};

export function MessageBubble({ role, content, pending }: Props) {
  const isUser = role === 'user';
  return (
    <View style={[styles.row, isUser ? styles.rowUser : styles.rowChar]}>
      <View
        style={[
          styles.bubble,
          isUser ? styles.bubbleUser : styles.bubbleChar,
          pending && styles.pending,
        ]}
      >
        <Text style={styles.text}>{content}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: { width: '100%', marginVertical: theme.spacing(0.5) },
  rowUser: { alignItems: 'flex-end' },
  rowChar: { alignItems: 'flex-start' },
  bubble: {
    maxWidth: '82%',
    paddingVertical: theme.spacing(1),
    paddingHorizontal: theme.spacing(1.5),
    borderRadius: theme.radius,
  },
  bubbleUser: { backgroundColor: theme.colors.bubbleUser, borderBottomRightRadius: 4 },
  bubbleChar: { backgroundColor: theme.colors.bubbleChar, borderBottomLeftRadius: 4 },
  pending: { opacity: 0.6 },
  text: { color: theme.colors.text, fontSize: 15, lineHeight: 21 },
});
