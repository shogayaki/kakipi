using UnityEngine;
using UnityEngine.SceneManagement;

/// <summary>
/// ゲーム全体の状態・スコア・コインを管理するシングルトン。
/// Step 11 で本実装予定。ここでは骨格のみ。
/// </summary>
public class GameManager : MonoBehaviour
{
    public static GameManager Instance { get; private set; }

    [Header("Game State")]
    public int Score { get; private set; }
    public int Coins { get; private set; }
    public int CurrentStage { get; private set; } = 1;

    // イベント（UIManager が購読する）
    public System.Action<int> OnScoreChanged;
    public System.Action<int> OnCoinChanged;

    void Awake()
    {
        if (Instance != null && Instance != this)
        {
            Destroy(gameObject);
            return;
        }
        Instance = this;
        DontDestroyOnLoad(gameObject);
    }

    public void AddScore(int amount)
    {
        Score += amount;
        OnScoreChanged?.Invoke(Score);
    }

    public void AddCoin(int amount = 1)
    {
        Coins += amount;
        OnCoinChanged?.Invoke(Coins);
    }

    public void LoadStage(int stageNumber)
    {
        CurrentStage = stageNumber;
        SceneManager.LoadScene($"Stage{stageNumber}");
    }

    public void GameOver()
    {
        SceneManager.LoadScene("GameOver");
    }

    public void ReturnToTitle()
    {
        Score = 0;
        Coins = 0;
        SceneManager.LoadScene("Title");
    }
}
