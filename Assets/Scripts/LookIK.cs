using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

[RequireComponent(typeof(Animator))]
public class LookIK : MonoBehaviour 
{
    [SerializeField]
    private GameObject lookObject;

	private Animator animator;

	void Start () 
	{
		animator = GetComponent<Animator>();
	}
	
	void Update () 
	{
		
	}

    void OnAnimatorIK(int layerIndex)
	{
		if(lookObject != null)
		{
			animator.SetLookAtWeight(1);
			animator.SetLookAtPosition(lookObject.transform.position);
		}
    }
}
