using UnityEngine;

/// <summary>
/// PlayerPrefs を使ったセーブ・ロードシステム。
/// Step 28 で本実装予定。
/// </summary>
public static class SaveSystem
{
    private const string KEY_HIGH_SCORE    = "HighScore";
    private const string KEY_UNLOCKED_STAGE = "UnlockedStage";
    private const string KEY_TOTAL_COINS   = "TotalCoins";
    private const string KEY_BGM_VOLUME    = "BGMVolume";
    private const string KEY_SFX_VOLUME    = "SFXVolume";

    public static void SaveHighScore(int score)
    {
        if (score > GetHighScore())
            PlayerPrefs.SetInt(KEY_HIGH_SCORE, score);
        PlayerPrefs.Save();
    }

    public static int GetHighScore() =>
        PlayerPrefs.GetInt(KEY_HIGH_SCORE, 0);

    public static void SaveUnlockedStage(int stage)
    {
        if (stage > GetUnlockedStage())
            PlayerPrefs.SetInt(KEY_UNLOCKED_STAGE, stage);
        PlayerPrefs.Save();
    }

    public static int GetUnlockedStage() =>
        PlayerPrefs.GetInt(KEY_UNLOCKED_STAGE, 1);

    public static void AddTotalCoins(int coins)
    {
        PlayerPrefs.SetInt(KEY_TOTAL_COINS, GetTotalCoins() + coins);
        PlayerPrefs.Save();
    }

    public static int GetTotalCoins() =>
        PlayerPrefs.GetInt(KEY_TOTAL_COINS, 0);

    public static void SaveVolume(float bgm, float sfx)
    {
        PlayerPrefs.SetFloat(KEY_BGM_VOLUME, bgm);
        PlayerPrefs.SetFloat(KEY_SFX_VOLUME, sfx);
        PlayerPrefs.Save();
    }

    public static float GetBGMVolume() =>
        PlayerPrefs.GetFloat(KEY_BGM_VOLUME, 1.0f);

    public static float GetSFXVolume() =>
        PlayerPrefs.GetFloat(KEY_SFX_VOLUME, 1.0f);

    public static void ResetAll()
    {
        PlayerPrefs.DeleteAll();
        PlayerPrefs.Save();
    }
}
