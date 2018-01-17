using UnityEngine;
using System.Collections;
using UnityEngine.UI;
using UnityEngine.SceneManagement;

public class GameManager : MonoBehaviour
{
	public int lives = 3;
	public int bricks = 28;
	public float resetDelay = 1f;
	public Text livesText;
	public GameObject gameOver;
	public GameObject youWon;
	public GameObject bricksPrefab;
	public GameObject paddle;
	public GameObject ball;
	public GameObject deathParticles;
	public static GameManager instance = null;

	private GameObject clonePaddle;
	private GameObject cloneBall;

	// Use this for initialization
	void Awake()
	{
		if (instance == null)
			instance = this;
		else if (instance != this)
			Destroy(gameObject);

		Setup();

	}

	private void Update()
	{
		CheckGameOver();
	}
	public void Setup()
	{
		clonePaddle = Instantiate(paddle, transform.position, Quaternion.identity) as GameObject;
		cloneBall = Instantiate(ball, new Vector3(transform.position.x, transform.position.y + 0.5f, transform.position.z), Quaternion.identity) as GameObject;
		cloneBall.transform.parent = clonePaddle.transform;
		Instantiate(bricksPrefab, transform.position, Quaternion.identity);
	}

	void CheckGameOver()
	{
		if (bricks < 1)
		{
			youWon.SetActive(true);
			Time.timeScale = 0.5f;
			Invoke("Reset", resetDelay);
		}

		if (lives < 1)
		{
			gameOver.SetActive(true);
			Time.timeScale = 0.5f;
			Invoke("Reset", resetDelay);
		}

	}

	void Reset()
	{
		Time.timeScale = 1f;
		WebcamHandDetection.webcamTexture.Stop();
		SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
	}

	public void LoseLife()
	{
		lives--;
		livesText.text = "Lives: " + lives;
		Instantiate(deathParticles, cloneBall.transform.position, Quaternion.identity);
		Destroy(clonePaddle);
		Destroy(cloneBall);
		Invoke("SetupPadlleBall", resetDelay);
	}

	void SetupPadlleBall()
	{
		clonePaddle = Instantiate(paddle, transform.position, Quaternion.identity) as GameObject;
		cloneBall = Instantiate(ball, new Vector3(transform.position.x, transform.position.y + 0.5f, transform.position.z), Quaternion.identity) as GameObject;
		cloneBall.transform.parent = clonePaddle.transform;
	}

	public void DestroyBrick()
	{
		bricks--;
		CheckGameOver();
	}
}