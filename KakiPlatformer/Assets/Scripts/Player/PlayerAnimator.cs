using UnityEngine;

/// <summary>
/// PlayerController の状態を読み取り、Animator パラメータを更新する。
/// Step 6 で Animator Controller と紐づける。
/// ここでは骨格のみ（パラメータ名は後で Animator Controller 作成時に合わせる）。
/// </summary>
[RequireComponent(typeof(Animator))]
[RequireComponent(typeof(PlayerController))]
public class PlayerAnimator : MonoBehaviour
{
    // Animator パラメータ名（Animator Controller と一致させること）
    private static readonly int PARAM_SPEED    = Animator.StringToHash("Speed");
    private static readonly int PARAM_GROUNDED = Animator.StringToHash("IsGrounded");
    private static readonly int PARAM_VELOC_Y  = Animator.StringToHash("VelocityY");
    private static readonly int PARAM_ATTACK   = Animator.StringToHash("Attack");
    private static readonly int PARAM_HURT     = Animator.StringToHash("Hurt");
    private static readonly int PARAM_DEAD     = Animator.StringToHash("IsDead");

    private Animator animator;
    private PlayerController controller;

    void Awake()
    {
        animator   = GetComponent<Animator>();
        controller = GetComponent<PlayerController>();
    }

    void Update()
    {
        animator.SetFloat(PARAM_SPEED,    Mathf.Abs(controller.Velocity.x));
        animator.SetBool (PARAM_GROUNDED, controller.IsGrounded);
        animator.SetFloat(PARAM_VELOC_Y,  controller.Velocity.y);
    }

    /// <summary>攻撃アニメーションをトリガー（PlayerAttack から呼ぶ）</summary>
    public void TriggerAttack()  => animator.SetTrigger(PARAM_ATTACK);

    /// <summary>被弾アニメーションをトリガー（HealthSystem から呼ぶ）</summary>
    public void TriggerHurt()    => animator.SetTrigger(PARAM_HURT);

    /// <summary>死亡アニメーションを開始（HealthSystem から呼ぶ）</summary>
    public void SetDead(bool val) => animator.SetBool(PARAM_DEAD, val);
}
