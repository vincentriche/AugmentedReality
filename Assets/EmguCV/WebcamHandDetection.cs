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


public class WebcamHandDetection : MonoBehaviour
{
	public static WebCamTexture webcamTexture;
	private static Texture2D resultTexture;
	private Color32[] background;
	private GCHandle backgroundHandle;
	private Mat backgroundMat;
	private Color32[] data;
	private Color32[] data2;
	private byte[] bytes;
	private byte[] grayBytes;
	private WebCamDevice[] devices;
	public int cameraCount = 0;
	private FlipType flip = FlipType.Horizontal;
	public bool hasBackground = false;
	public bool drawDetection = true;
	public GameObject screen;

	public static bool handDetected = false;
	public static Vector2 barycenter = Vector2.zero;
	public static List<Vector2> detectedFingers = new List<Vector2>();

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
		if (hasBackground == false && Input.GetKeyDown(KeyCode.Space))
		{
			background = new Color32[webcamTexture.width * webcamTexture.height];
			webcamTexture.GetPixels32(background);
			backgroundHandle = GCHandle.Alloc(background, GCHandleType.Pinned);
			backgroundMat = new Mat(new Size(webcamTexture.width, webcamTexture.height), DepthType.Cv8U, 4, backgroundHandle.AddrOfPinnedObject(), webcamTexture.width * 4);
			hasBackground = true;
			return;
		}

		if (hasBackground == true && webcamTexture != null && webcamTexture.didUpdateThisFrame)
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
			GCHandle grayHandle = GCHandle.Alloc(grayBytes, GCHandleType.Pinned);
			GCHandle resultHandle = GCHandle.Alloc(bytes, GCHandleType.Pinned);

			Mat currentWebcamMat = new Mat(new Size(webcamTexture.width, webcamTexture.height), DepthType.Cv8U, 4, handle.AddrOfPinnedObject(), webcamTexture.width * 4);
			Mat resultMat = new Mat(webcamTexture.height, webcamTexture.width, DepthType.Cv8U, 3, resultHandle.AddrOfPinnedObject(), webcamTexture.width * 3);
			Mat maskMat = new Mat(webcamTexture.height, webcamTexture.width, DepthType.Cv8U, 1, grayHandle.AddrOfPinnedObject(), webcamTexture.width * 1);

			#region do some image processing here
			CvInvoke.CvtColor(currentWebcamMat, resultMat, ColorConversion.Bgra2Bgr);

			CvInvoke.MedianBlur(currentWebcamMat, currentWebcamMat, 7);
			CvInvoke.AbsDiff(backgroundMat, currentWebcamMat, currentWebcamMat);
			CvInvoke.Threshold(currentWebcamMat, currentWebcamMat, 40, 255, ThresholdType.Binary);
			CvInvoke.CvtColor(currentWebcamMat, maskMat, ColorConversion.Bgra2Gray);
			CvInvoke.Threshold(maskMat, maskMat, 1, 255, ThresholdType.Binary);
			DetectHandContour(maskMat, resultMat, drawDetection);
			#endregion

			if (flip != FlipType.None)
				CvInvoke.Flip(resultMat, resultMat, flip);

			handle.Free();
			resultHandle.Free();

			// DISPLAY
			if (resultTexture == null || resultTexture.width != webcamTexture.width ||
				resultTexture.height != webcamTexture.height)
			{
				resultTexture = new Texture2D(webcamTexture.width, webcamTexture.height, TextureFormat.RGB24, false);
			}

			resultTexture.LoadRawTextureData(bytes);
			resultTexture.Apply();
		}
	}

	void OnGUI()
	{
		if (drawDetection && resultTexture != null)
		{
			//UnityEngine.Graphics.DrawTexture(new Rect(0, 0, UnityEngine.Screen.width, UnityEngine.Screen.height), resultTexture);
			screen.GetComponent<MeshRenderer>().sharedMaterial.SetTexture("_MainTex", resultTexture);
		}
	}

	public static void DetectHandContour(Mat image, Mat result, bool draw)
	{
		handDetected = false;
		int largest_contour_index = 0;
		double largest_area = 0;

		VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
		Mat hierarchy = new Mat();

		CvInvoke.FindContours(image, contours, hierarchy, RetrType.Ccomp, ChainApproxMethod.ChainApproxSimple);

		for (int i = 0; i < contours.Size; i++)
		{
			double a = CvInvoke.ContourArea(contours[i], false); // Find the area of contour
			if (a > largest_area)
			{
				largest_area = a;
				largest_contour_index = i; //Store the index of largest contour
			}

		}
		if (draw == true)
			CvInvoke.DrawContours(result, contours, largest_contour_index, new MCvScalar(255, 0, 0));

		if (contours.Size <= 0 || contours[largest_contour_index].Size <= 0)
			return;
		handDetected = true;

		VectorOfPoint contourLargest = contours[largest_contour_index];
		PointF[] vf = new PointF[contourLargest.Size];
		for (int i = 0; i < contourLargest.Size; i++)
			vf[i] = new PointF(contourLargest[i].X, contourLargest[i].Y);
		VectorOfPointF contourLargestFloat = new VectorOfPointF(vf);
		VectorOfPointF hullLargestFloat = new VectorOfPointF();
		VectorOfInt hullLargestInt = new VectorOfInt();

		CvInvoke.ConvexHull(contourLargestFloat, hullLargestFloat, false, true);
		CvInvoke.ConvexHull(contourLargestFloat, hullLargestInt, false, true);

		Point[] v = new Point[hullLargestFloat.Size];
		for (int i = 0; i < hullLargestFloat.Size; i++)
			v[i] = new Point((int)hullLargestFloat[i].X, (int)hullLargestFloat[i].Y);
		VectorOfPoint hullLargest = new VectorOfPoint(v);
		if (draw == true)
			CvInvoke.Polylines(result, hullLargest, true, new MCvScalar(255, 0, 0));

		if (hullLargestInt.Size <= 3)
			return;

		VectorOfRect defects = new VectorOfRect();
		CvInvoke.ConvexityDefects(contourLargest, hullLargestInt, defects);
		List<int> validDefects = new List<int>();
		for (int i = 1; i < defects.Size; i++)
		{
			if (defects[i].Height > 4000)
				validDefects.Add(i);
		}

		// Draw convex defects
		if (draw == true)
			for (int i = 0; i < validDefects.Count; i++)
			{
				Point farthestPoint = contourLargest[defects[validDefects[i]].Width];
				Point beginPoint = contourLargest[defects[validDefects[i]].X];
				Point endPoint = contourLargest[defects[validDefects[i]].Y];

				CvInvoke.Circle(result, farthestPoint, 3, new Bgr(System.Drawing.Color.Red).MCvScalar, 2);
				CvInvoke.Line(result, beginPoint, farthestPoint, new Bgr(System.Drawing.Color.Green).MCvScalar);
				CvInvoke.Line(result, farthestPoint, endPoint, new Bgr(System.Drawing.Color.Green).MCvScalar);
			}

		// Compute and draw barycenter
		PointF barycenterTemp = GetCentroid(hullLargestFloat);
		barycenter = new Vector2(barycenterTemp.X, barycenterTemp.Y);
		if (draw == true)
			CvInvoke.Circle(result, new Point((int)barycenter.x, (int)barycenter.y), 3, new Bgr(System.Drawing.Color.White).MCvScalar, 2);
		float polyMaxSize = GetLargestPolyDiagonal(hullLargestFloat);

		// Create list of finger tips
		detectedFingers = new List<Vector2>();
		for (int i = 0; i < validDefects.Count; i++)
		{
			Vector2 fingerPoint = new Vector2(contourLargest[defects[validDefects[i]].X].X, contourLargest[defects[validDefects[i]].X].Y);
			Vector2 fingerPoint2 = new Vector2(contourLargest[defects[validDefects[i]].Y].X, contourLargest[defects[validDefects[i]].Y].Y);

			if (fingerPoint.y > barycenter.y / 2.0f)
			{
				bool flag = true;
				for (int j = 0; j < detectedFingers.Count; j++)
					if (Vector2.Distance(fingerPoint, detectedFingers[j]) < polyMaxSize / 6)
					{
						flag = false;
						break;
					}
				if (flag == true)
					detectedFingers.Add(fingerPoint);
			}
			if (fingerPoint2.y > barycenter.y / 2.0f)
			{
				bool flag = true;
				for (int j = 0; j < detectedFingers.Count; j++)
					if (Vector2.Distance(fingerPoint2, detectedFingers[j]) < polyMaxSize / 6)
					{
						flag = false;
						break;
					}
				if (flag == true)
					detectedFingers.Add(fingerPoint2);
			}
		}

		// Draw finger tips
		if (draw == true)
			for (int i = 0; i < detectedFingers.Count; i++)
			{
				CvInvoke.Circle(result, new Point((int)detectedFingers[i].x, (int)detectedFingers[i].y), 3, new Bgr(System.Drawing.Color.Green).MCvScalar, 2);
			}
	}

	public static PointF GetCentroid(VectorOfPointF poly)
	{
		float accumulatedArea = 0.0f;
		float centerX = 0.0f;
		float centerY = 0.0f;

		for (int i = 0, j = poly.Size - 1; i < poly.Size; j = i++)
		{
			float temp = poly[i].X * poly[j].Y - poly[j].X * poly[i].Y;
			accumulatedArea += temp;
			centerX += (poly[i].X + poly[j].X) * temp;
			centerY += (poly[i].Y + poly[j].Y) * temp;
		}

		if (Math.Abs(accumulatedArea) < 1E-7f)
			return PointF.Empty;  // Avoid division by zero

		accumulatedArea *= 3f;
		return new PointF(centerX / accumulatedArea, centerY / accumulatedArea);
	}

	public static float GetLargestPolyDiagonal(VectorOfPointF poly)
	{
		float maxLength = 0.0f;
		for (int i = 0; i < poly.Size; i++)
		{
			for (int j = 0; j < poly.Size; j++)
			{
				maxLength = Mathf.Max(maxLength, PointDistance(poly[i], poly[j]));
			}
		}
		return maxLength;
	}

	private static float PointDistance(PointF a, PointF b)
	{
		float xd = b.X - a.X;
		float yd = b.Y - a.Y;
		return Mathf.Sqrt(xd * xd + yd * yd);
	}

	public static Vector2 GetNormalizedHandPosition()
	{
		if (resultTexture == null)
			return Vector2.zero;

		Vector2 temp = new Vector2(barycenter.x / (float)resultTexture.width, barycenter.y / (float)resultTexture.height);
		temp.x = 1.0f - temp.x;
		return temp;
	}

	public static List<Vector2> GetNormalizedFingerPositions()
	{
		if (resultTexture == null)
			return null;

		List<Vector2> res = new List<Vector2>(detectedFingers);
		for(int i = 0; i < res.Count; i++)
		{
			Vector2 temp = new Vector2(res[i].x / (float)resultTexture.width, res[i].y / (float)resultTexture.height);
			temp.x = 1.0f - temp.x;
			res[i] = temp;
		}
		return res;
	}
}
