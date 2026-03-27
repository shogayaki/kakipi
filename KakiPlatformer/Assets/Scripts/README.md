# Scripts フォルダ構成

## Player/
- `PlayerController.cs` - 移動・ジャンプ制御 (Step 2-3)
- `PlayerAttack.cs` - 攻撃処理 (Step 7)
- `PlayerAnimator.cs` - アニメーション制御 (Step 6)

## Enemy/
- `EnemyBase.cs` - 敵の基底クラス (Step 9)
- `SlimeEnemy.cs` - スライム（往復歩行）(Step 9)
- `JumpEnemy.cs` - ジャンプ型敵 (Step 19)
- `FlyingEnemy.cs` - 飛行型敵 (Step 20)
- `BossEnemy.cs` - ボス (Step 21)

## System/
- `GameManager.cs` - ゲーム状態・スコア管理 (Step 11)
- `AudioManager.cs` - BGM/SE管理 (Step 24)
- `SaveSystem.cs` - セーブ・ロード (Step 28)
- `HealthSystem.cs` - HP管理 (Step 8)
- `SceneLoader.cs` - シーン遷移 (Step 16)

## UI/
- `UIManager.cs` - HUD更新 (Step 12)
- `TitleScreen.cs` - タイトル画面 (Step 17)
- `GameOverScreen.cs` - ゲームオーバー (Step 16)
- `StageClearScreen.cs` - クリア画面 (Step 15)
- `BossHealthBar.cs` - ボスHPバー (Step 21)

## Items/
- `Coin.cs` - コイン収集 (Step 11)
- `PowerUpBase.cs` - パワーアップ基底 (Step 23)
- `SpeedUpItem.cs` - 速度UP (Step 23)
- `InvincibleItem.cs` - 無敵 (Step 23)
- `HealthItem.cs` - HP回復 (Step 23)
