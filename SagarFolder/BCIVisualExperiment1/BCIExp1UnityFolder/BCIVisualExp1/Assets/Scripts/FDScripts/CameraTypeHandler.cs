using UnityEngine;
using System.Collections;
using InControl;


public class CameraTypeHandler : MonoBehaviour 
{

	public GameObject normalCameraObject;
	public GameObject ovrCameraControllerObject;

	public Transform lookAtTarget;
	public bool isOVR = false;
	public float ovrFOV = 120.0f;

	Transform meshCreatureTransform;

	Vector3 lerpedUpVector;
	float lerpScale = 1.0f;

	// Use this for initialization
	void Start () 
	{
		normalCameraObject.SetActive(!isOVR);
		ovrCameraControllerObject.SetActive(isOVR);

		ovrCameraControllerObject.GetComponent<OVRCameraController>().SetVerticalFOV(ovrFOV);

		ovrCameraControllerObject.transform.GetChild(0).camera.fieldOfView = ovrFOV;
		ovrCameraControllerObject.transform.GetChild(1).camera.fieldOfView = ovrFOV;

		meshCreatureTransform = GameObject.FindGameObjectWithTag("MeshCreature").transform;

		lerpedUpVector = meshCreatureTransform.transform.up;
	}
	
	// Update is called once per frame
	void Update () 
	{

		var inputDevice = InputManager.ActiveDevice;		

		if(inputDevice.LeftBumper.WasPressed)
		{
			isOVR = !isOVR;
			
			normalCameraObject.SetActive(!isOVR);
			ovrCameraControllerObject.SetActive(isOVR);

		}

		lerpedUpVector = Vector3.Lerp(lerpedUpVector, meshCreatureTransform.up, lerpScale * Time.deltaTime);

		normalCameraObject.transform.LookAt(lookAtTarget, lerpedUpVector);
		ovrCameraControllerObject.transform.LookAt(lookAtTarget, meshCreatureTransform.up);
	
	}
}
