using UnityEngine;
using System.Collections;
using System.Linq;

public class BCIDataDirector : MonoBehaviour 
{

	BCIStreamingDataEmulator streamingDataEmulator;

	void Awake()
	{
		streamingDataEmulator = FindObjectOfType<BCIStreamingDataEmulator>();
	}


	void Update()
	{

		Debug.Log("Latest data: " + streamingDataEmulator.GetDataPointsArray()[0] );
	}

}
