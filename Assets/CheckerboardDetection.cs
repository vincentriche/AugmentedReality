using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.Util;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.Face;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;


public class CheckerboardDetection : MonoBehaviour
{
	public static WebCamTexture webcamTexture;
	private static Texture2D displayTexture;
	private Color32[] data;
	private byte[] bytes;
	private FlipType flip = FlipType.Horizontal;
	public GameObject screen;

	void Start()
	{
		WebCamDevice[] devices = WebCamTexture.devices;
		int cameraCount = devices.Length;

		if (cameraCount > 0)
		{
			webcamTexture = new WebCamTexture(devices[0].name);
			webcamTexture.Play();
		}
	}

	void Update()
	{
		if (webcamTexture != null && webcamTexture.didUpdateThisFrame)
		{
			if (data == null || (data.Length != webcamTexture.width * webcamTexture.height))
				data = new Color32[webcamTexture.width * webcamTexture.height];

			webcamTexture.GetPixels32(data);
			//data = webcamTexture.GetPixels32(0);

			if (bytes == null || bytes.Length != data.Length * 3)
				bytes = new byte[data.Length * 3];

			// OPENCV PROCESSING
			GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
			GCHandle resultHandle = GCHandle.Alloc(bytes, GCHandleType.Pinned);

			Mat currentWebcamMat = new Mat(new Size(webcamTexture.width, webcamTexture.height), DepthType.Cv8U, 4, handle.AddrOfPinnedObject(), webcamTexture.width * 4);
			Mat resultMat = new Mat(webcamTexture.height, webcamTexture.width, DepthType.Cv8U, 3, resultHandle.AddrOfPinnedObject(), webcamTexture.width * 3);
			
			CvInvoke.CvtColor(currentWebcamMat, resultMat, ColorConversion.Bgra2Bgr);

			if (flip != FlipType.None)
				CvInvoke.Flip(resultMat, resultMat, flip);

			handle.Free();
			resultHandle.Free();

			// DISPLAY
			if (displayTexture == null || displayTexture.width != webcamTexture.width ||
				displayTexture.height != webcamTexture.height)
			{
				displayTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
			}

			displayTexture.LoadRawTextureData(bytes);
			displayTexture.Apply();
		}

		if (displayTexture != null)
		{
			screen.GetComponent<MeshRenderer>().sharedMaterial.SetTexture("_MainTex", displayTexture);
		}
	}
}
