﻿using UnityEngine;
using UnityEngine.UI;

public class PlayMovieTextureOnUI : MonoBehaviour
{
	public RawImage rawimage;

	void Start()
	{
		WebCamTexture webcamTexture = new WebCamTexture();
		rawimage.texture = webcamTexture;
		rawimage.material.mainTexture = webcamTexture;
		webcamTexture.Play();
	}
}