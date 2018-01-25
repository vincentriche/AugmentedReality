using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.UI;
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
	public RawImage rawImageDisplay;
	public int camNumber = 0;
	public Camera targetCamera;
	public GameObject target;
	public float patternScale = 1.0f;
	public Vector2 pattern = new Vector2(7, 4);
	public Vector2 requestedResolution = new Vector2(640, 480);
	private Size patternSize;
	private MCvTermCriteria criteria = new MCvTermCriteria(100, 1e-5);
	public static WebCamTexture webcamTexture;
	private static Texture2D displayTexture;
	private Color32[] data;
	private byte[] bytes;
	private byte[] grayBytes;
	private FlipType flip = FlipType.None;
	
	private Matrix<float> cvImageCorners;
	private Matrix<double> cvWorldCorners;
	private Matrix<double> cvIntrinsicParams;
	private Matrix<double> cvDistortionParams;


	void Start()
	{
		WebCamDevice[] devices = WebCamTexture.devices;
		int cameraCount = devices.Length;

		if (cameraCount > 0)
		{
			webcamTexture = new WebCamTexture(devices[camNumber].name, (int)requestedResolution.x, (int)requestedResolution.y, 60);
			webcamTexture.Play();
		}

		// Set target scale
		patternSize = new Size((int)pattern.x, (int)pattern.y);
		target.transform.localScale = new Vector3(patternScale * (patternSize.Width + 1), patternScale * (patternSize.Height + 1), 1.0f);

		// Construct world corner points
		Vector2 offset = new Vector2(patternSize.Width / 2.0f * patternScale, patternSize.Height / 2.0f * patternScale);
		cvWorldCorners = new Matrix<double>(patternSize.Height * patternSize.Width, 1, 3);
		for (int iy = 0; iy < patternSize.Height; iy++)
		{
			for (int ix = 0; ix < patternSize.Width; ix++)
			{
				cvWorldCorners.Data[iy * patternSize.Width + ix, 0] = ix * patternScale - offset.x;
				cvWorldCorners.Data[iy * patternSize.Width + ix, 1] = iy * patternScale - offset.y;
				cvWorldCorners.Data[iy * patternSize.Width + ix, 2] = 0;
			}
		}

		// Initialize intrinsic parameters
		cvIntrinsicParams = new Matrix<double>(3, 3, 1);
		cvIntrinsicParams[0, 0] = 1.2306403943428504e+03f;
		cvIntrinsicParams[0, 1] = 0;
		cvIntrinsicParams[0, 2] = (double)webcamTexture.width / 2.0d;
		cvIntrinsicParams[1, 0] = 0;
		cvIntrinsicParams[1, 1] = 1.2306403943428504e+03f;
		cvIntrinsicParams[1, 2] = (double)webcamTexture.height / 2.0d;
		cvIntrinsicParams[2, 0] = 0;
		cvIntrinsicParams[2, 1] = 0;
		cvIntrinsicParams[2, 2] = 1;

		cvDistortionParams = new Matrix<double>(4, 1, 1);
		cvDistortionParams[0, 0] = 1.9920531921963049e-02f;
		cvDistortionParams[1, 0] = 3.2143454945024297e-02f;
		cvDistortionParams[2, 0] = 0.0f;
		cvDistortionParams[3, 0] = 0.0f;
	}

	void OnDestroy()
	{
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
			
			cvImageCorners = new Matrix<float>(patternSize.Width * patternSize.Height, 1, 2);
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
			rawImageDisplay.texture = displayTexture;
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
		Matrix<float>[] split = cvImageCorners.Split();
		Matrix<double> doubleCvImageCorners = new Matrix<double>(patternSize.Height * patternSize.Width, 1, 2);
		for (int iy = 0; iy < patternSize.Height; iy++)
		{
			for (int ix = 0; ix < patternSize.Width; ix++)
			{
				doubleCvImageCorners.Data[iy * patternSize.Width + ix, 0] = split[0][iy * patternSize.Width + ix, 0];
				doubleCvImageCorners.Data[iy * patternSize.Width + ix, 1] = split[1][iy * patternSize.Width + ix, 0];
			}
		}

		// Compute the rotation / translation of the chessboard (the cameras extrinsic pramaters)
		Mat tempRotation = new Mat(3, 1, DepthType.Cv64F, 1);
		Mat translationMatrix = new Mat(3, 1, DepthType.Cv64F, 1);
		bool res = CvInvoke.SolvePnP(cvWorldCorners, cvImageCorners, cvIntrinsicParams, cvDistortionParams, tempRotation, translationMatrix);
		if (res == false)
			return;

		// Converte the rotation from 3 axis rotations into a rotation matrix.
		double[] tempRotationData = new double[3];
		Marshal.Copy(tempRotation.DataPointer, tempRotationData, 0, tempRotation.Width * tempRotation.Height);
		tempRotationData[0] = tempRotationData[0];
		tempRotationData[1] = -tempRotationData[1];
		tempRotationData[2] = -tempRotationData[2];
		GCHandle tempHandle = GCHandle.Alloc(tempRotationData, GCHandleType.Pinned);
		tempRotation = new Mat(3, 1, DepthType.Cv64F, 1, tempHandle.AddrOfPinnedObject(), sizeof(double));
		Mat rotationMatrix = new Mat(3, 3, DepthType.Cv64F, 1);
		CvInvoke.Rodrigues(tempRotation, rotationMatrix);

		double[] rotationData = new double[9];
		Marshal.Copy(rotationMatrix.DataPointer, rotationData, 0, rotationMatrix.Width * rotationMatrix.Height);
		double[] translationData = new double[3];
		Marshal.Copy(translationMatrix.DataPointer, translationData, 0, translationMatrix.Width * translationMatrix.Height);

		// Turn the intrinsic and extrinsic pramaters into the projection and model/view matrix
		Matrix4x4 projection = new Matrix4x4();
		projection.m00 = (float)(2 * cvIntrinsicParams[0, 0] / (double)webcamTexture.width);
		projection.m10 = 0;
		projection.m20 = 0;
		projection.m30 = 0;

		projection.m01 = 0;
		projection.m11 = (float)(2 * cvIntrinsicParams[1, 1] / (double)webcamTexture.height);
		projection.m21 = 0;
		projection.m31 = 0;

		projection.m02 = (float)(1 - 2 * cvIntrinsicParams[0, 2] / (double)webcamTexture.width);
		projection.m12 = (float)(-1 + (2 * cvIntrinsicParams[1, 2] + 2) / (double)webcamTexture.height);
		projection.m22 = (targetCamera.nearClipPlane + targetCamera.farClipPlane) / (targetCamera.nearClipPlane - targetCamera.farClipPlane);
		projection.m32 = -1;

		projection.m03 = 0;
		projection.m13 = 0;
		projection.m23 = 2 * (targetCamera.nearClipPlane * targetCamera.farClipPlane) / (targetCamera.nearClipPlane - targetCamera.farClipPlane);
		projection.m33 = 0;

		targetCamera.projectionMatrix = projection;
		targetCamera.fieldOfView = Mathf.Atan(1.0f / projection.m11) * 2.0f * Mathf.Rad2Deg;
		targetCamera.aspect = (float)webcamTexture.width / (float)webcamTexture.height;


		Matrix4x4 cvModelView = new Matrix4x4();
		cvModelView.m00 = (float)rotationData[0 * 3 + 0];
		cvModelView.m10 = (float)rotationData[1 * 3 + 0];
		cvModelView.m20 = (float)rotationData[2 * 3 + 0];
		cvModelView.m30 = 0;

		cvModelView.m01 = (float)rotationData[0 * 3 + 1];
		cvModelView.m11 = (float)rotationData[1 * 3 + 1];
		cvModelView.m21 = (float)rotationData[2 * 3 + 1];
		cvModelView.m31 = 0;

		cvModelView.m02 = (float)rotationData[0 * 3 + 2];
		cvModelView.m12 = (float)rotationData[1 * 3 + 2];
		cvModelView.m22 = (float)rotationData[2 * 3 + 2];
		cvModelView.m32 = 0;

		cvModelView.m03 = (float)translationData[0];
		cvModelView.m13 = (float)translationData[1];
		cvModelView.m23 = (float)translationData[2];
		cvModelView.m33 = 1;
		cvModelView = cvModelView.inverse;

		/*Vector3 camPos = Vector3.zero;
		Quaternion camRot = Quaternion.identity;
		ExtractCameraTRS(cvModelView, out camPos, out camRot);*/
		//cvModelView = InvertModelView(cvModelView);
		targetCamera.transform.position = ExtractPositionOpenCV(cvModelView);
		targetCamera.transform.rotation = ExtractRotationOpenCV(cvModelView);
	}

	public Quaternion ExtractRotationOpenCV(Matrix4x4 matrix)
	{
		Vector3 forward;
		forward.x = matrix.m02;
		forward.y = -matrix.m12;
		forward.z = matrix.m22;

		Vector3 upwards;
		upwards.x = matrix.m01;
		upwards.y = -matrix.m11;
		upwards.z = matrix.m21;

		return Quaternion.LookRotation(forward, upwards);
	}

	public Vector3 ExtractPositionOpenCV(Matrix4x4 matrix)
	{
		Vector3 position;
		position.x = matrix.m03;
		position.y = -matrix.m13;
		position.z = matrix.m23;
		return position;
	}

	/*public void ExtractCameraTRS(Matrix4x4 matrix, out Vector3 pos, out Quaternion rot)
	{
		Matrix4x4 rotation = matrix;
		rotation.m03 = 0;
		rotation.m13 = 0;
		rotation.m23 = 0;
		rotation = rotation.transpose;
		Vector4 temp = rotation.GetRow(1);
		rotation.SetRow(1, rotation.GetRow(2));
		rotation.SetRow(2, temp);

		Vector3 forward;
		forward.x = rotation.m02;
		forward.y = rotation.m12;
		forward.z = rotation.m22;
		Vector3 upwards;
		upwards.x = rotation.m01;
		upwards.y = rotation.m11;
		upwards.z = rotation.m21;
		rot = Quaternion.LookRotation(forward, upwards);
		rot = rot * Quaternion.AngleAxis(90, Vector3.right);
		rotation = Matrix4x4.TRS(Vector3.zero, rot, Vector3.one);

		Vector3 position;
		position.x = matrix.m03;
		position.y = matrix.m13;
		position.z = matrix.m23;

		Matrix4x4 negativeMatrix = Matrix4x4.identity;
		negativeMatrix.m00 = -1;
		negativeMatrix.m11 = -1;
		negativeMatrix.m22 = -1;
		negativeMatrix.m33 = -1;
		pos = negativeMatrix * rotation * position;
	}*/

	public Matrix4x4 InvertModelView(Matrix4x4 matrix)
	{
		Matrix4x4 negativeMatrix = Matrix4x4.identity;
		negativeMatrix.m00 = -1;
		negativeMatrix.m11 = -1;
		negativeMatrix.m22 = -1;
		negativeMatrix.m33 = -1;

		Matrix4x4 rotation = matrix;
		rotation.m03 = 0;
		rotation.m13 = 0;
		rotation.m23 = 0;
		rotation = rotation.transpose;

		Vector3 position;
		position.x = matrix.m03;
		position.y = matrix.m13;
		position.z = matrix.m23;
		position = (negativeMatrix * rotation) * position;

		Matrix4x4 result;
		result.m00 = rotation.m00;
		result.m10 = rotation.m10;
		result.m20 = rotation.m20;
		result.m30 = 0;

		result.m01 = rotation.m01;
		result.m11 = rotation.m11;
		result.m21 = rotation.m21;
		result.m31 = 0;

		result.m02 = rotation.m02;
		result.m12 = rotation.m12;
		result.m22 = rotation.m22;
		result.m32 = 0;

		result.m03 = position.x;
		result.m13 = position.y;
		result.m23 = position.z;
		result.m33 = 1;
		return result;

		//return matrix.inverse;
	}
}
