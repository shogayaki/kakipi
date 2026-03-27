using UnityEngine;

/// <summary>
/// BGM・SE を一元管理するシングルトン。
/// Step 24-25 で本実装予定。ここでは骨格のみ。
/// </summary>
public class AudioManager : MonoBehaviour
{
    public static AudioManager Instance { get; private set; }

    [Header("Audio Sources")]
    [SerializeField] private AudioSource bgmSource;
    [SerializeField] private AudioSource sfxSource;

    [Header("BGM Clips")]
    public AudioClip bgmTitle;
    public AudioClip bgmStage1;
    public AudioClip bgmStage2;
    public AudioClip bgmStage3;
    public AudioClip bgmBoss;
    public AudioClip bgmGameOver;
    public AudioClip bgmClear;

    [Header("SFX Clips")]
    public AudioClip sfxJump;
    public AudioClip sfxDoubleJump;
    public AudioClip sfxAttack;
    public AudioClip sfxHit;
    public AudioClip sfxDamage;
    public AudioClip sfxCoin;
    public AudioClip sfxItem;
    public AudioClip sfxDeath;
    public AudioClip sfxClear;

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

    public void PlayBGM(AudioClip clip)
    {
        if (bgmSource.clip == clip && bgmSource.isPlaying) return;
        bgmSource.clip = clip;
        bgmSource.loop = true;
        bgmSource.Play();
    }

    public void StopBGM()
    {
        bgmSource.Stop();
    }

    public void PlaySFX(AudioClip clip)
    {
        if (clip == null) return;
        sfxSource.PlayOneShot(clip);
    }

    public void SetBGMVolume(float volume)
    {
        bgmSource.volume = volume;
    }

    public void SetSFXVolume(float volume)
    {
        sfxSource.volume = volume;
    }
}
