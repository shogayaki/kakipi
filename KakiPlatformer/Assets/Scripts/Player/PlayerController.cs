using UnityEngine;
using UnityEngine.InputSystem;

/// <summary>
/// プレイヤーの移動・ジャンプを管理するコントローラー。
/// Step 2: 左右移動・地面判定
/// Step 3: ジャンプ・二段ジャンプ（次ステップで拡張）
/// </summary>
[RequireComponent(typeof(Rigidbody2D))]
[RequireComponent(typeof(CapsuleCollider2D))]
public class PlayerController : MonoBehaviour
{
    // ---- 移動パラメータ ----
    [Header("Movement")]
    [SerializeField] private float moveSpeed = 5.0f;
    [SerializeField] private float jumpForce = 12.0f;
    [SerializeField] private float fallMultiplier = 2.5f;   // 落下加速係数
    [SerializeField] private float lowJumpMultiplier = 2.0f; // ボタン離し時の急落下係数

    // ---- 地面判定 ----
    [Header("Ground Check")]
    [SerializeField] private Transform groundCheckPoint;
    [SerializeField] private float groundCheckRadius = 0.15f;
    [SerializeField] private LayerMask groundLayer;

    // ---- 二段ジャンプ ----
    [Header("Double Jump")]
    [SerializeField] private bool enableDoubleJump = true;

    // ---- 内部状態 ----
    private Rigidbody2D rb;
    private Vector2 moveInput;
    private bool isGrounded;
    private bool hasDoubleJump;
    private bool jumpPressed;
    private bool isFacingRight = true;

    // 外部参照用プロパティ
    public bool IsGrounded => isGrounded;
    public bool IsFacingRight => isFacingRight;
    public Vector2 Velocity => rb.linearVelocity;
    public bool IsMoving => Mathf.Abs(moveInput.x) > 0.01f;

    // ---- Unity ライフサイクル ----

    void Awake()
    {
        rb = GetComponent<Rigidbody2D>();
        rb.freezeRotation = true; // 物理回転を防ぐ
    }

    void Update()
    {
        CheckGround();
        HandleFlip();
        ApplyFallMultiplier();
    }

    void FixedUpdate()
    {
        Move();
    }

    // ---- 地面判定 ----

    private void CheckGround()
    {
        bool wasGrounded = isGrounded;
        isGrounded = Physics2D.OverlapCircle(
            groundCheckPoint.position,
            groundCheckRadius,
            groundLayer
        );

        // 着地した瞬間に二段ジャンプをリセット
        if (!wasGrounded && isGrounded)
        {
            hasDoubleJump = enableDoubleJump;
        }
    }

    // ---- 移動 ----

    private void Move()
    {
        rb.linearVelocity = new Vector2(moveInput.x * moveSpeed, rb.linearVelocity.y);
    }

    // ---- 左右反転 ----

    private void HandleFlip()
    {
        if (moveInput.x > 0.01f && !isFacingRight)
            Flip();
        else if (moveInput.x < -0.01f && isFacingRight)
            Flip();
    }

    private void Flip()
    {
        isFacingRight = !isFacingRight;
        Vector3 scale = transform.localScale;
        scale.x *= -1;
        transform.localScale = scale;
    }

    // ---- 落下加速 ----

    private void ApplyFallMultiplier()
    {
        if (rb.linearVelocity.y < 0)
        {
            // 落下中は重力を強める
            rb.linearVelocity += Vector2.up * Physics2D.gravity.y
                * (fallMultiplier - 1) * Time.deltaTime;
        }
        else if (rb.linearVelocity.y > 0 && !jumpPressed)
        {
            // ジャンプボタンを離したら急落下
            rb.linearVelocity += Vector2.up * Physics2D.gravity.y
                * (lowJumpMultiplier - 1) * Time.deltaTime;
        }
    }

    // ---- Input System コールバック ----

    /// <summary>Move アクションのコールバック（Input System）</summary>
    public void OnMove(InputAction.CallbackContext context)
    {
        moveInput = context.ReadValue<Vector2>();
    }

    /// <summary>Jump アクションのコールバック（Input System）</summary>
    public void OnJump(InputAction.CallbackContext context)
    {
        if (context.started)
        {
            jumpPressed = true;
            TryJump();
        }
        else if (context.canceled)
        {
            jumpPressed = false;
        }
    }

    // ---- ジャンプ処理 ----

    private void TryJump()
    {
        if (isGrounded)
        {
            PerformJump();
        }
        else if (enableDoubleJump && hasDoubleJump)
        {
            hasDoubleJump = false;
            PerformJump();
        }
    }

    private void PerformJump()
    {
        // Y速度をリセットしてから力を加える（二段ジャンプでも一定の高さを保証）
        rb.linearVelocity = new Vector2(rb.linearVelocity.x, 0f);
        rb.AddForce(Vector2.up * jumpForce, ForceMode2D.Impulse);
    }

    // ---- 外部からの制御 ----

    /// <summary>ノックバック（ダメージ時に呼ぶ）</summary>
    public void ApplyKnockback(Vector2 direction, float force)
    {
        rb.linearVelocity = Vector2.zero;
        rb.AddForce(direction * force, ForceMode2D.Impulse);
    }

    /// <summary>移動を一時ロック（演出・死亡時等）</summary>
    public void SetMovementLocked(bool locked)
    {
        if (locked)
        {
            moveInput = Vector2.zero;
            rb.linearVelocity = Vector2.zero;
        }
    }

    // ---- Gizmos（エディタ表示）----

    void OnDrawGizmosSelected()
    {
        if (groundCheckPoint == null) return;
        Gizmos.color = isGrounded ? Color.green : Color.red;
        Gizmos.DrawWireSphere(groundCheckPoint.position, groundCheckRadius);
    }
}
