using UnityEngine;
using System.Collections;

public class TimedDestroy : MonoBehaviour
{
    public float destroyTime = 1f;
    
    void Start()
    {
        Destroy(gameObject, destroyTime);
    }

}