using UnityEngine;
using System;

/// <summary>
/// プレイヤー・敵共用のHP管理コンポーネント。
/// Step 8 で使用する。
/// </summary>
public class HealthSystem : MonoBehaviour
{
    [SerializeField] private int maxHP = 3;

    public int CurrentHP { get; private set; }
    public int MaxHP => maxHP;
    public bool IsAlive => CurrentHP > 0;

    // イベント
    public event Action<int, int> OnHPChanged; // (currentHP, maxHP)
    public event Action OnDeath;

    void Awake()
    {
        CurrentHP = maxHP;
    }

    public void TakeDamage(int amount)
    {
        if (!IsAlive) return;

        CurrentHP = Mathf.Max(0, CurrentHP - amount);
        OnHPChanged?.Invoke(CurrentHP, maxHP);

        if (CurrentHP == 0)
            OnDeath?.Invoke();
    }

    public void Heal(int amount)
    {
        CurrentHP = Mathf.Min(maxHP, CurrentHP + amount);
        OnHPChanged?.Invoke(CurrentHP, maxHP);
    }

    public void SetMaxHP(int newMax)
    {
        maxHP = newMax;
        CurrentHP = Mathf.Min(CurrentHP, maxHP);
        OnHPChanged?.Invoke(CurrentHP, maxHP);
    }
}
