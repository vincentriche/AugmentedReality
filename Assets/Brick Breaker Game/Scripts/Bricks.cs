using UnityEngine;
using System.Collections;

public class Bricks : MonoBehaviour
{
    public GameObject brickParticle;

    void OnCollisionEnter(Collision other)
    {
        GameManager.instance.bricks--;
        Instantiate(brickParticle, transform.position, Quaternion.identity);
        Destroy(gameObject);
    }
}