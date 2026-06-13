import { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  FlatList,
  KeyboardAvoidingView,
  Platform,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { sendChat, type ChatMessage } from '../api/chat';
import { type Character } from '../characters';
import { MessageBubble } from '../components/MessageBubble';
import { StaminaBar } from '../components/StaminaBar';
import { type UseStamina } from '../stamina';
import { theme } from '../theme';

type Props = {
  character: Character;
  stamina: UseStamina;
  onBack: () => void;
};

type DisplayMessage = ChatMessage & { id: string; error?: boolean };

// サーバーに渡す会話履歴は直近Nターンに絞ってトークン節約する
const HISTORY_LIMIT = 12;

export function ChatScreen({ character, stamina, onBack }: Props) {
  const [messages, setMessages] = useState<DisplayMessage[]>([
    { id: 'greeting', role: 'assistant', content: character.greeting },
  ]);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const listRef = useRef<FlatList<DisplayMessage>>(null);

  useEffect(() => {
    const t = setTimeout(() => listRef.current?.scrollToEnd({ animated: true }), 50);
    return () => clearTimeout(t);
  }, [messages, sending]);

  const handleSend = async () => {
    const text = input.trim();
    if (!text || sending) return;

    if (!stamina.consume()) {
      setMessages((m) => [
        ...m,
        {
          id: `sys-${Date.now()}`,
          role: 'assistant',
          content: 'スタミナが足りません。広告を見て回復するか、時間をおいてね。',
          error: true,
        },
      ]);
      return;
    }

    const userMsg: DisplayMessage = { id: `u-${Date.now()}`, role: 'user', content: text };
    const next = [...messages, userMsg];
    setMessages(next);
    setInput('');
    setSending(true);

    const history: ChatMessage[] = next
      .filter((m) => m.id !== 'greeting' && !m.error)
      .slice(-HISTORY_LIMIT)
      .map(({ role, content }) => ({ role, content }));

    const result = await sendChat(character.id, history);
    setSending(false);
    setMessages((m) => [
      ...m,
      result.ok
        ? { id: `a-${Date.now()}`, role: 'assistant', content: result.reply }
        : { id: `e-${Date.now()}`, role: 'assistant', content: result.error, error: true },
    ]);
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : undefined}
    >
      <View style={[styles.header, { borderBottomColor: character.color }]}>
        <TouchableOpacity onPress={onBack} hitSlop={12}>
          <Text style={styles.back}>‹ 戻る</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>
          {character.emoji} {character.name}
        </Text>
        <View style={{ width: 48 }} />
      </View>

      <View style={styles.staminaWrap}>
        <StaminaBar
          value={stamina.value}
          max={stamina.max}
          secondsToNext={stamina.secondsToNext}
          adsRemaining={stamina.adsRemaining}
          onWatchAd={stamina.watchAd}
        />
      </View>

      <FlatList
        ref={listRef}
        style={styles.list}
        contentContainerStyle={styles.listContent}
        data={messages}
        keyExtractor={(m) => m.id}
        renderItem={({ item }) => (
          <MessageBubble role={item.role} content={item.content} />
        )}
        ListFooterComponent={
          sending ? (
            <View style={styles.typing}>
              <ActivityIndicator color={theme.colors.primary} />
              <Text style={styles.typingText}>{character.name} が入力中…</Text>
            </View>
          ) : null
        }
      />

      <View style={styles.inputBar}>
        <TextInput
          style={styles.input}
          value={input}
          onChangeText={setInput}
          placeholder={stamina.canChat ? 'メッセージを入力…' : 'スタミナ切れです'}
          placeholderTextColor={theme.colors.textMuted}
          editable={!sending}
          multiline
          onSubmitEditing={handleSend}
        />
        <TouchableOpacity
          style={[styles.sendBtn, (sending || !input.trim()) && styles.sendBtnDisabled]}
          onPress={handleSend}
          disabled={sending || !input.trim()}
        >
          <Text style={styles.sendBtnText}>送信</Text>
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: theme.colors.bg },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: theme.spacing(6),
    paddingBottom: theme.spacing(1.5),
    paddingHorizontal: theme.spacing(2),
    borderBottomWidth: 2,
  },
  back: { color: theme.colors.textMuted, fontSize: 16, width: 48 },
  headerTitle: { color: theme.colors.text, fontSize: 18, fontWeight: '700' },
  staminaWrap: { padding: theme.spacing(1.5) },
  list: { flex: 1 },
  listContent: { padding: theme.spacing(2) },
  typing: { flexDirection: 'row', alignItems: 'center', gap: 8, padding: theme.spacing(1) },
  typingText: { color: theme.colors.textMuted, fontSize: 13 },
  inputBar: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    padding: theme.spacing(1),
    gap: theme.spacing(1),
    backgroundColor: theme.colors.surface,
  },
  input: {
    flex: 1,
    maxHeight: 120,
    color: theme.colors.text,
    backgroundColor: theme.colors.surfaceAlt,
    borderRadius: theme.radius,
    paddingHorizontal: theme.spacing(1.5),
    paddingVertical: theme.spacing(1),
    fontSize: 15,
  },
  sendBtn: {
    backgroundColor: theme.colors.primary,
    borderRadius: theme.radius,
    paddingHorizontal: theme.spacing(2),
    paddingVertical: theme.spacing(1.25),
  },
  sendBtnDisabled: { backgroundColor: theme.colors.surfaceAlt },
  sendBtnText: { color: theme.colors.text, fontWeight: '700' },
});
