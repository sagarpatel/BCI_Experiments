using UnityEngine;
using System.Collections;

public class BCIAudioGenerator : MonoBehaviour 
{
	BCIDataDirector bciDataDirector;

	public float gain = 0.1f;

	void Awake()
	{

		bciDataDirector = FindObjectOfType<BCIDataDirector>();

	}

	void OnAudioFilterRead(float[] data, int channels)
	{


		float[] bciData = bciDataDirector.currentDataArray;
		int bciDataLength = bciData.Length;
		int audioDataLength = data.Length/channels;


		// assuming that bciDataLength is always smaller than data length
		int sampleRatio = audioDataLength/bciDataLength ;
		for(int i  = 0; i < data.Length ; i  = i + channels)
		{
			int theoreticalBCILocation = i / sampleRatio;

			if(theoreticalBCILocation >= bciData.Length)
				break;



			data[i] = gain * bciData[theoreticalBCILocation];

			// copy data to both channels if exist
			if(channels == 2)
			{
				data[i +1] = data[i];
			}
		}



	}


}
