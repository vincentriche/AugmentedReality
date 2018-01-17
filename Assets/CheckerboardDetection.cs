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
	private byte[] grayBytes;
	private FlipType flip = FlipType.Horizontal;
	private Size patternSize = new Size(7, 4);
	private MCvTermCriteria criteria = new MCvTermCriteria(100, 1e-5);
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
			if (grayBytes == null || grayBytes.Length != data.Length * 1)
				grayBytes = new byte[data.Length * 1];

			// OPENCV PROCESSING
			GCHandle handle = GCHandle.Alloc(data, GCHandleType.Pinned);
			GCHandle resultHandle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
			GCHandle grayHandle = GCHandle.Alloc(grayBytes, GCHandleType.Pinned);

			Mat currentWebcamMat = new Mat(new Size(webcamTexture.width, webcamTexture.height), DepthType.Cv8U, 4, handle.AddrOfPinnedObject(), webcamTexture.width * 4);
			Mat resultMat = new Mat(webcamTexture.height, webcamTexture.width, DepthType.Cv8U, 3, resultHandle.AddrOfPinnedObject(), webcamTexture.width * 3);
			Mat grayMat = new Mat(webcamTexture.height, webcamTexture.width, DepthType.Cv8U, 1, grayHandle.AddrOfPinnedObject(), webcamTexture.width * 1);

			CvInvoke.CvtColor(currentWebcamMat, resultMat, ColorConversion.Bgra2Bgr);
			CvInvoke.CvtColor(resultMat, grayMat, ColorConversion.Bgra2Gray);
			VectorOfPoint cornerPoints = DetectCheckerboard(grayMat);
			if (cornerPoints.Size > 0)
				DrawCheckerboard(resultMat, cornerPoints);

			handle.Free();
			resultHandle.Free();
			grayHandle.Free();

			if (flip != FlipType.None)
				CvInvoke.Flip(resultMat, resultMat, flip);
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

	private VectorOfPoint DetectCheckerboard(Mat image)
	{
		VectorOfPoint cornerPoints = new VectorOfPoint();
		bool result = CvInvoke.FindChessboardCorners(image, patternSize, cornerPoints);

		if (result == false)
			return new VectorOfPoint();

		Debug.Log(result);
		return cornerPoints;
	}

	private void DrawCheckerboard(Mat image, VectorOfPoint cornerPoints)
	{
		CvInvoke.CornerSubPix(image, cornerPoints, new Size(5, 5), new Size(-1, -1), criteria);
		CvInvoke.DrawChessboardCorners(image, patternSize, cornerPoints, true);
	}
}
