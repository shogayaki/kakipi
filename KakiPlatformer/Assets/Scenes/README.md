# シーン一覧

Unity の Build Settings に以下の順で登録すること:

| Index | Scene 名 | 内容 |
|---|---|---|
| 0 | Title | タイトル・メインメニュー |
| 1 | Stage1 | ステージ1（草原の丘） |
| 2 | Stage2 | ステージ2（暗黒の洞窟） |
| 3 | Stage3 | ステージ3（天空の城）+ ボス |
| 4 | GameOver | ゲームオーバー画面 |
| 5 | GameClear | ゲームクリア画面 |

## 各シーンに必要な GameObjects

### 全シーン共通
- `GameManager` (GameManager.cs)
- `AudioManager` (AudioManager.cs)
- `SceneLoader` (SceneLoader.cs) ※フェード用Canvas含む

### プレイシーン（Stage1〜3）
- `Player` (PlayerController, PlayerAttack, PlayerAnimator, HealthSystem)
- `CinemachineVirtualCamera`
- `UIManager` (UIManager.cs)
- `Tilemap` (Ground レイヤー)
- `DeathZone`

### タイトルシーン
- `TitleScreen` (TitleScreen.cs)
