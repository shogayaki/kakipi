using UnityEngine;

/// <summary>
/// プレイヤーがステージ外（穴）に落ちたときに即死させるゾーン。
/// Trigger コライダーを持つ GameObject にアタッチする。
/// </summary>
public class DeathZone : MonoBehaviour
{
    void OnTriggerEnter2D(Collider2D other)
    {
        if (!other.CompareTag("Player")) return;

        var health = other.GetComponent<HealthSystem>();
        if (health != null)
        {
            // HP を 0 にして死亡イベントを発火
            health.TakeDamage(health.MaxHP);
        }
        else
        {
            // HealthSystem がない場合の保険
            GameManager.Instance?.GameOver();
        }
    }
}
