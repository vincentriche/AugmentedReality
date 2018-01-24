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
	public GameObject screen;
	public Camera targetCamera;
	private Size patternSize = new Size(7, 4);
	private float patternScale = 1.0f;
	private MCvTermCriteria criteria = new MCvTermCriteria(100, 1e-5);
	public static WebCamTexture webcamTexture;
	private static Texture2D displayTexture;
	private Color32[] data;
	private byte[] bytes;
	private byte[] grayBytes;
	private FlipType flip = FlipType.Horizontal;

	private PointF[] imageCorners;
	private GCHandle imageCornersHandle;
	private Matrix<float> cvImageCorners;

	private MCvPoint3D32f[] worldCornerPoints;
	private GCHandle worldCornersHandle;
	private Matrix<float> cvWorldCorners;

	private float[] intrinsicParams;
	private GCHandle intrinsicParamsHandle;
	private Matrix<float> cvIntrinsicParams;

	private float[] distortionParams;
	private GCHandle distortionParamsHandle;
	private Matrix<float> cvDistortionParams;


	void Start()
	{
		WebCamDevice[] devices = WebCamTexture.devices;
		int cameraCount = devices.Length;

		if (cameraCount > 0)
		{
			webcamTexture = new WebCamTexture(devices[0].name);
			webcamTexture.Play();
		}
		
		// Construct world corner points
		worldCornerPoints = new MCvPoint3D32f[patternSize.Height * patternSize.Width];
		for (int ix = 0; ix < patternSize.Height; ix++)
			for (int iy = 0; iy < patternSize.Width; iy++)
				worldCornerPoints[iy * patternSize.Height + ix] = new MCvPoint3D32f(iy * patternScale, ix * patternScale, 0);
		worldCornersHandle = GCHandle.Alloc(worldCornerPoints, GCHandleType.Pinned);
		cvWorldCorners = new Matrix<float>(worldCornerPoints.Length, 1, 3, worldCornersHandle.AddrOfPinnedObject(), 3 * sizeof(float));

		imageCorners = new PointF[patternSize.Width * patternSize.Height];
		imageCornersHandle = GCHandle.Alloc(imageCorners, GCHandleType.Pinned);
		cvImageCorners = new Matrix<float>(imageCorners.Length, 1, 2, imageCornersHandle.AddrOfPinnedObject(), 2 * sizeof(float));

		// Initialize intrinsic parameters
		intrinsicParams = new float[9];
		intrinsicParams[0] = 1.2306403943428504e+03F;
		intrinsicParams[1] = 0;
		intrinsicParams[2] = 640;
		intrinsicParams[3] = 0;
		intrinsicParams[4] = 1.2306403943428504e+03F;
		intrinsicParams[5] = 480;
		intrinsicParams[6] = 0;
		intrinsicParams[7] = 0;
		intrinsicParams[8] = 1;
		/*intrinsicParams[0] = 1.2306403943428504e+03F;
		intrinsicParams[3] = 0;
		intrinsicParams[6] = 640;
		intrinsicParams[1] = 0;
		intrinsicParams[4] = 1.2306403943428504e+03F;
		intrinsicParams[7] = 480;
		intrinsicParams[2] = 0;
		intrinsicParams[5] = 0;
		intrinsicParams[8] = 1;*/
		intrinsicParamsHandle = GCHandle.Alloc(intrinsicParams, GCHandleType.Pinned);
		cvIntrinsicParams = new Matrix<float>(3, 3, 1, intrinsicParamsHandle.AddrOfPinnedObject(), sizeof(float) * 3);

		distortionParams = new float[4];
		distortionParams[0] = 1.9920531921963049e-02F;
		distortionParams[1] = 3.2143454945024297e-02F;
		distortionParams[2] = 0;
		distortionParams[3] = 0;
		distortionParamsHandle = GCHandle.Alloc(distortionParams, GCHandleType.Pinned);
		cvDistortionParams = new Matrix<float>(distortionParams.Length, 1, 1, distortionParamsHandle.AddrOfPinnedObject(), sizeof(float));
	}

	void OnDestroy()
	{
		imageCornersHandle.Free();
		worldCornersHandle.Free();
		intrinsicParamsHandle.Free();
		distortionParamsHandle.Free();
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
			
			bool detected = DetectCheckerboard(grayMat, resultMat);
			if (detected == true)
				SetCameraTransformFromChessboard();

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

	private bool DetectCheckerboard(Mat detectImage, Mat drawImage = null)
	{
		bool result = CvInvoke.FindChessboardCorners(detectImage, patternSize, cvImageCorners);

		if (result == false)
			return false;

		CvInvoke.CornerSubPix(detectImage, cvImageCorners, new Size(5, 5), new Size(-1, -1), criteria);
		
		if (drawImage != null)
			CvInvoke.DrawChessboardCorners(drawImage, patternSize, cvImageCorners, true);

		return true;
	}

	private void SetCameraTransformFromChessboard()
	{
		float[] rotData = new float[3];
		GCHandle rotHandle = GCHandle.Alloc(rotData, GCHandleType.Pinned);
		Matrix<float> tempRotation = new Matrix<float>(3, 1, 1, rotHandle.AddrOfPinnedObject(), sizeof(float));
		float[] posData = new float[3];
		GCHandle posHandle = GCHandle.Alloc(rotData, GCHandleType.Pinned);
		Matrix<float> translationMatrix = new Matrix<float>(3, 1, 1, posHandle.AddrOfPinnedObject(), sizeof(float));

		/*Matrix<float> cvIntrinsicParams2 = new Matrix<float>(3, 3);
		Matrix<float> cvDistortionParams2 = new Matrix<float>(4, 1);
		Matrix<float> cvImageCorners2 = new Matrix<float>(patternSize.Height * patternSize.Width, 2);
		Matrix<float> cvWorldCorners2 = new Matrix<float>(patternSize.Height * patternSize.Width, 3);
		Matrix<float> tempRotation = new Matrix<float>(3, 1);
		Matrix<float> translationMatrix = new Matrix<float>(3, 1);*/

		// Compute the rotation / translation of the chessboard (the cameras extrinsic pramaters)
		CvInvoke.SolvePnP(cvWorldCorners, cvImageCorners, cvIntrinsicParams, cvDistortionParams, tempRotation, translationMatrix);

		// Converte the rotation from 3 axis rotations into a rotation matrix.
		Matrix<float> rotationMatrix = new Matrix<float>(3, 3);
		CvInvoke.Rodrigues(tempRotation, rotationMatrix);

		// Turn the intrinsic and extrinsic pramaters into the projection and model/view matrix
		/*Matrix4x4 projection = new Matrix4x4();
		projection.m00 = 2 * calibrationMatrix[0, 0] / 640.0f;
		projection.m01 = 0;
		projection.m02 = 0;
		projection.m03 = 0;

		projection.m10 = 0;
		projection.m11 = 2 * calibrationMatrix[1, 1] / 480.0f;
		projection.m12 = 0;
		projection.m13 = 0;

		projection.m20 = 1 - 2 * calibrationMatrix[0, 2] / 640.0f;
		projection.m21 = -1 + (2 * calibrationMatrix[1, 2] + 2) / 480.0f;
		projection.m22 = (targetCamera.nearClipPlane + targetCamera.farClipPlane) / (targetCamera.nearClipPlane - targetCamera.farClipPlane);
		projection.m23 = -1;

		projection.m30 = 0;
		projection.m31 = 0;
		projection.m32 = 2 * (targetCamera.nearClipPlane * targetCamera.farClipPlane) / (targetCamera.nearClipPlane - targetCamera.farClipPlane); ;
		projection.m33 = 0;


		Matrix4x4 cameraTRS = new Matrix4x4();
		cameraTRS.m00 = rotationMatrix[0, 0];
		cameraTRS.m01 = rotationMatrix[1, 0];
		cameraTRS.m02 = rotationMatrix[2, 0];
		cameraTRS.m03 = 0;

		cameraTRS.m10 = rotationMatrix[0, 1];
		cameraTRS.m11 = rotationMatrix[1, 1];
		cameraTRS.m12 = rotationMatrix[2, 1];
		cameraTRS.m13 = 0;

		cameraTRS.m20 = rotationMatrix[0, 2];
		cameraTRS.m21 = rotationMatrix[1, 2];
		cameraTRS.m22 = rotationMatrix[2, 2];
		cameraTRS.m23 = 0;

		cameraTRS.m30 = translationMatrix[0, 0];
		cameraTRS.m31 = translationMatrix[1, 0];
		cameraTRS.m32 = translationMatrix[2, 0];
		cameraTRS.m33 = 1;

		targetCamera.transform.position = ExtractPosition(cameraTRS);
		targetCamera.transform.rotation = ExtractRotation(cameraTRS);
		targetCamera.projectionMatrix = projection;*/
	}

	public Quaternion ExtractRotation(Matrix4x4 matrix)
	{
		Vector3 forward;
		forward.x = matrix.m02;
		forward.y = matrix.m12;
		forward.z = matrix.m22;

		Vector3 upwards;
		upwards.x = matrix.m01;
		upwards.y = matrix.m11;
		upwards.z = matrix.m21;

		return Quaternion.LookRotation(forward, upwards);
	}

	public Vector3 ExtractPosition(Matrix4x4 matrix)
	{
		Vector3 position;
		position.x = matrix.m03;
		position.y = matrix.m13;
		position.z = matrix.m23;
		return position;
	}

	public Vector3 ExtractScale(Matrix4x4 matrix)
	{
		Vector3 scale;
		scale.x = new Vector4(matrix.m00, matrix.m10, matrix.m20, matrix.m30).magnitude;
		scale.y = new Vector4(matrix.m01, matrix.m11, matrix.m21, matrix.m31).magnitude;
		scale.z = new Vector4(matrix.m02, matrix.m12, matrix.m22, matrix.m32).magnitude;
		return scale;
	}
}
