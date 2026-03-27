# Player Prefab セットアップ手順

Unity エディタで以下の手順でプレイヤー Prefab を作成する。

---

## 1. GameObject 構成

```
Player (GameObject)
├── Sprite (子GameObject)
│   └── SpriteRenderer
└── GroundCheck (子GameObject)  ← 足元に配置
└── AttackPoint (子GameObject)  ← 前方に配置
```

---

## 2. Player GameObject のコンポーネント設定

### Rigidbody2D
| 項目 | 値 |
|---|---|
| Body Type | Dynamic |
| Gravity Scale | 3.0 |
| Freeze Rotation Z | ✓ |
| Collision Detection | Continuous |

### CapsuleCollider2D
| 項目 | 値 |
|---|---|
| Direction | Vertical |
| Size | (0.5, 1.0) |
| Offset | (0, 0) |

### PlayerInput（Input System コンポーネント）
| 項目 | 値 |
|---|---|
| Actions | PlayerInputActions（Assets/Scripts/Player/フォルダ） |
| Behavior | **Invoke Unity Events** |

#### Events の設定（PlayerInput Inspector）
- `Player / Move` → `PlayerController.OnMove`
- `Player / Jump` → `PlayerController.OnJump`
- `Player / Attack` → `PlayerAttack.OnAttack`

### PlayerController
| フィールド | 値 |
|---|---|
| Move Speed | 5.0 |
| Jump Force | 12.0 |
| Fall Multiplier | 2.5 |
| Low Jump Multiplier | 2.0 |
| Ground Check Point | GroundCheck (子Object) |
| Ground Check Radius | 0.15 |
| Ground Layer | Ground レイヤー |
| Enable Double Jump | ✓ |

### PlayerAnimator
（Step 6 で Animator Controller を作成してから設定する）

### PlayerAttack
| フィールド | 値 |
|---|---|
| Attack Damage | 1 |
| Attack Cooldown | 0.4 |
| Attack Range | 1.2 |
| Attack Point | AttackPoint (子Object) |
| Enemy Layer | Enemy レイヤー |

### HealthSystem
| フィールド | 値 |
|---|---|
| Max HP | 5 |

---

## 3. 子 GameObject の配置

### GroundCheck
- Position: (0, -0.55, 0)  ← コライダー底面の少し下

### AttackPoint
- Position: (0.7, 0, 0)  ← 右向き時の前方
- ※ 左右反転は PlayerController.Flip() が `localScale.x` を反転するため自動的に反転される

---

## 4. レイヤー設定

Unity メニュー `Edit → Project Settings → Tags and Layers` で追加：

| Layer番号 | 名前 |
|---|---|
| 6 | Ground |
| 7 | Player |
| 8 | Enemy |
| 9 | Item |
| 10 | PlayerAttack |
| 11 | EnemyAttack |

### Physics2D レイヤー衝突設定
`Edit → Project Settings → Physics2D` の Layer Collision Matrix：
- Player と Enemy : 衝突する（ON）
- PlayerAttack と Enemy : 衝突する（ON）
- EnemyAttack と Player : 衝突する（ON）
- PlayerAttack と Ground : 衝突しない（OFF）

---

## 5. Prefab 保存
完成したら `Assets/Prefabs/Player/Player.prefab` として保存。
