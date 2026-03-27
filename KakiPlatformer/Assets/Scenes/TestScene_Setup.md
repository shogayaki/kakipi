# Step 2 動作確認用テストシーン

## 最小構成で動作確認する手順

### 1. 新規シーン作成
`File → New Scene → Basic 2D` → `TestMovement` という名前で保存

### 2. 必要な GameObject を配置

#### カメラ
- Main Camera の `Projection` を `Orthographic` に設定
- `Size` = 5

#### 地面
1. `GameObject → 2D Object → Sprite → Square` を作成
2. 名前: `Ground`
3. Transform: Position (0, -3, 0), Scale (20, 1, 1)
4. Layer: `Ground`
5. `BoxCollider2D` を追加（デフォルトで付いている）
6. SpriteRenderer の Color を緑色に

#### プレイヤー
1. `Assets/Prefabs/Player/Player.prefab` をシーンにドラッグ
   （または手動で Player_Setup.md の手順で作成）
2. Position: (0, 0, 0)

#### DeathZone
1. `GameObject → Create Empty` → 名前: `DeathZone`
2. Transform: Position (0, -10, 0), Scale (100, 1, 1)
3. `BoxCollider2D` 追加 → `Is Trigger` を ON
4. `DeathZone.cs` をアタッチ

### 3. 確認項目

| チェック | 内容 |
|---|---|
| ☐ | A/D キー（または ←/→）で左右に移動する |
| ☐ | 地面に立っている（落下しない） |
| ☐ | 右を向いているとき D キー、左を向いているとき A キーで Sprite が反転する |
| ☐ | Scene View の Player 足元に緑の OverlapCircle が表示される（IsGrounded=true） |
| ☐ | DeathZone に落ちても即座にクラッシュしない（GameManager があれば GameOver に遷移） |

### 4. 次ステップ（Step 3）で追加確認
| チェック | 内容 |
|---|---|
| ☐ | スペースキーでジャンプする |
| ☐ | 空中でもう一度スペースで二段ジャンプする |
| ☐ | ボタンを早く離すと低くジャンプする（可変ジャンプ） |
| ☐ | 落下が自然に加速される |
