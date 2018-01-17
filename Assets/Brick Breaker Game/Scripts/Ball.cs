using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class Ball : MonoBehaviour
{
	public float ballInitialVelocity = 200f;
	public float bumperForce = 4.0f;

	private Rigidbody rg;
	private bool ballInPlay;
	private bool touchingPaddle = true;

	void Awake()
	{
		rg = GetComponent<Rigidbody>();
	}

	void Update()
	{
		List<Vector2> fingerPositions = WebcamHandDetection.GetNormalizedFingerPositions();
		if (fingerPositions != null && fingerPositions.Count >= 3)
		{
			ballInPlay = true;
			if (touchingPaddle == true)
			{
				transform.parent = null;
				//transform.localScale = new Vector3(sphereScale, sphereScale, sphereScale);
				rg.isKinematic = false;
				rg.AddForce(new Vector3(ballInitialVelocity, ballInitialVelocity, 0));
				touchingPaddle = false;
			}
		}

		if (fingerPositions != null && fingerPositions.Count < 3)
		{
			ballInPlay = false;
		}
	}

	void OnCollisionEnter(Collision other)
	{
		if (other.gameObject.CompareTag("Player") == true && ballInPlay == false)
		{
			transform.parent = other.transform;
			//transform.localScale = new Vector3(sphereScale, sphereScale, sphereScale);
			//transform.localScale = new Vector3(sphereScale / other.transform.localScale.x, sphereScale / other.transform.localScale.y, sphereScale / other.transform.localScale.z);
			rg.isKinematic = true;
			rg.velocity = Vector3.zero;
			touchingPaddle = true;
		}
		else
		{
			rg.AddForce(other.contacts[0].normal * bumperForce, ForceMode.Impulse);
		}
	}
}