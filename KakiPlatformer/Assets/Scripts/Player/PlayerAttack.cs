using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// プレイヤーの近接攻撃を管理する。
/// Step 7 で本実装（攻撃判定・クールダウン）。
/// Step 2 では骨格のみ。
/// </summary>
public class PlayerAttack : MonoBehaviour
{
    [Header("Attack Settings")]
    [SerializeField] private int attackDamage = 1;
    [SerializeField] private float attackCooldown = 0.4f;
    [SerializeField] private float attackRange = 1.2f;
    [SerializeField] private Transform attackPoint;
    [SerializeField] private LayerMask enemyLayer;

    private bool canAttack = true;
    private PlayerAnimator playerAnimator;

    void Awake()
    {
        playerAnimator = GetComponent<PlayerAnimator>();
    }

    /// <summary>Attack アクションのコールバック（Input System）</summary>
    public void OnAttack(InputAction.CallbackContext context)
    {
        if (context.started && canAttack)
            StartCoroutine(AttackRoutine());
    }

    private IEnumerator AttackRoutine()
    {
        canAttack = false;
        playerAnimator?.TriggerAttack();

        // 攻撃判定（アニメーションの中間タイミングで実行）
        yield return new WaitForSeconds(0.1f);
        PerformAttackHit();

        yield return new WaitForSeconds(attackCooldown - 0.1f);
        canAttack = true;
    }

    private void PerformAttackHit()
    {
        if (attackPoint == null) return;

        Collider2D[] hits = Physics2D.OverlapCircleAll(
            attackPoint.position, attackRange, enemyLayer);

        foreach (var hit in hits)
        {
            var health = hit.GetComponent<HealthSystem>();
            health?.TakeDamage(attackDamage);
        }
    }

    void OnDrawGizmosSelected()
    {
        if (attackPoint == null) return;
        Gizmos.color = Color.yellow;
        Gizmos.DrawWireSphere(attackPoint.position, attackRange);
    }
}
