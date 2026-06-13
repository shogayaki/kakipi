import { StatusBar } from 'expo-status-bar';
import { useState } from 'react';
import { SafeAreaView, StyleSheet } from 'react-native';
import { type Character } from './src/characters';
import { CharacterSelectScreen } from './src/screens/CharacterSelectScreen';
import { ChatScreen } from './src/screens/ChatScreen';
import { useStamina } from './src/stamina';
import { theme } from './src/theme';

// MVPなので外部ルーターは使わず、選択状態だけで画面を切り替える軽量ナビゲーション。
export default function App() {
  const [active, setActive] = useState<Character | null>(null);
  // スタミナはアプリ全体で共有する(画面を行き来しても保持)
  const stamina = useStamina();

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="light" />
      {active ? (
        <ChatScreen
          character={active}
          stamina={stamina}
          onBack={() => setActive(null)}
        />
      ) : (
        <CharacterSelectScreen onSelect={setActive} />
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: theme.colors.bg },
});
