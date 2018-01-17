using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Paddle : MonoBehaviour
{
	public float paddleSpeed = 1f;
	private Vector3 playerPos = new Vector3(0, -2.62f, 0);
	float previousRot = 0.0f;
	float previousPosX = 0.0f;

	void Update()
	{
		Vector2 handPos = WebcamHandDetection.GetNormalizedHandPosition();
		float leftPos = Camera.main.transform.position.x - Camera.main.orthographicSize;
		float xPos = leftPos + handPos.x * Camera.main.orthographicSize * 2.0f;
		float newPosX = Mathf.Lerp(previousPosX, xPos, 0.2f);
		previousPosX = newPosX;
		playerPos = new Vector3(Mathf.Clamp(newPosX, -4.3f, 4.3f), -2.62f, 0f);
		transform.position = playerPos;

		List<Vector2> fingerPositions = WebcamHandDetection.GetNormalizedFingerPositions();
		if(fingerPositions != null && fingerPositions.Count > 0)
		{
			float rot = 0.1f;
			for (int i = 0; i < fingerPositions.Count; i++)
			{
				rot += fingerPositions[i].x - handPos.x;
			}
			float newRot = Mathf.Lerp(previousRot, rot, 0.2f);
			transform.rotation = Quaternion.Euler(0.0f, 0.0f, -newRot * 60.0f);
			previousRot = newRot;
		}
	}
}