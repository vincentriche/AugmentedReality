using UnityEngine;
using System.Collections;

public class DeadZone : MonoBehaviour
{
	void OnTriggerEnter(Collider col)
	{
		if (col.gameObject.CompareTag("Ball") == true)
			GameManager.instance.LoseLife();
	}
}