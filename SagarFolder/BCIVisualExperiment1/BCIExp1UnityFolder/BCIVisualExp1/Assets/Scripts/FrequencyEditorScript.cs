using UnityEngine;
using System.Collections;

public class FrequencyEditorScript : MonoBehaviour 
{

	int minIndex = -1;
	int maxIndex = 9;
	public int currentIndex;

	float inputCooldown = 0.2f;
	float cooldownCounter = 0;

	float incrementCooldown = 0.2f;
	float incrementCooldownCounter = 0;
	float incrementHeldDownDurationCounter = 0;

	public GameObject rangeMarker;
	Vector3 rangeMarkerPosition = new Vector3();

	public GUISkin guiSkin;
	public bool isActive = false;

	int minSamples = 1;
	int maxSamples = 512;

	int minSampleStartIndex = 0;
	int maxSampleStartIndex = 1000;

	Color markerColor = new Color(0,1,0, 0.7f);

	AudioDirectorScript audioDirector;
	GeneralEditorScript generalEditor;


	// Use this for initialization
	void Start () 
	{
		currentIndex = minIndex;
		rangeMarker = (GameObject)Instantiate(rangeMarker, new Vector3(), Quaternion.identity);

		audioDirector =  (AudioDirectorScript)GameObject.FindWithTag("AudioDirector").GetComponent("AudioDirectorScript");
		generalEditor = (GeneralEditorScript)GetComponent("GeneralEditorScript");

	}
	
	// Update is called once per frame
	void Update () 
	{

		if(isActive && generalEditor.isActive)
		{
			HandleInputs();

			foreach (Transform child in rangeMarker.transform)
			{
				child.gameObject.renderer.enabled = true;
				child.gameObject.renderer.material.color = markerColor;
			}
			
			rangeMarkerPosition = new Vector3(0,0, 40 * currentIndex );
			rangeMarker.transform.position = rangeMarkerPosition;
			AdjustRangeMarkerScale();
		}
		else
		{
			foreach (Transform child in rangeMarker.transform)
				child.gameObject.renderer.enabled = false;
		}


	
	}

	void OnGUI() 
 	{
 		if(isActive && generalEditor.isActive)
 		{
 			if( currentIndex != -1)
 			{
 				// Calculate frequency range of current decade
 				float minFrequency = 0;
 				float maxFrequency = 0;
 				CalculateDecadeFrequencyRange(currentIndex, out minFrequency, out maxFrequency);

	    		GUI.Label(new Rect(0.0f, 0.05f*Screen.height, Screen.width, 0.2f*Screen.height), "Current Decade Frequency Range: " + minFrequency.ToString() + "Hz to " + maxFrequency.ToString() + "Hz", guiSkin.label );
	    		GUI.Label(new Rect(0.0f, 0.1f*Screen.height, Screen.width, 0.2f*Screen.height), "Samples in Current Decade: " + audioDirector.samplesPerDecadeArray[currentIndex].ToString(), guiSkin.label );
	    	}
	    	else
	    	{
	    		float sampleCount = (float)audioDirector.sampleArrayFreqBH.Length;
				float sampleRate = (float)audioDirector.currentSampleRate;

				float startFrequency = ( (float)audioDirector.sampleStartIndex * sampleRate/2.0f )/sampleCount;
	    		GUI.Label(new Rect(0.0f, 0.05f*Screen.height, Screen.width, 0.2f*Screen.height), "Start Frequency: " + startFrequency.ToString() + "Hz", guiSkin.label );
	    	}

    	}
    }


	void HandleInputs()
	{

		// handle range selection
		if( Input.GetAxis("Editor Horizontal") != 0)
		{
			if( cooldownCounter > inputCooldown )
			{
				// actually apply input
				if( Input.GetAxis("Editor Horizontal") > 0)
					currentIndex += 1;
				else if ( Input.GetAxis("Editor Horizontal") < 0)
					currentIndex -= 1;

				cooldownCounter = 0;

			}
			else
				cooldownCounter += Time.deltaTime;

		}
		else
			cooldownCounter += Time.deltaTime;


		if( currentIndex < minIndex )
			currentIndex = minIndex;
		else if( currentIndex > maxIndex )
			currentIndex = maxIndex;			


		// handle incrementing
		if( Input.GetAxis("Editor Vertical") != 0)
		{	

			if( incrementCooldownCounter > (incrementCooldown - incrementHeldDownDurationCounter/10.0f) )
			{
				// smaple count stuff
				if( currentIndex != -1)
				{
					int tempSampleCount = audioDirector.samplesPerDecadeArray[currentIndex];
					
					if( Input.GetAxis("Editor Vertical") > 0)
						tempSampleCount += 1;
					else if( Input.GetAxis("Editor Vertical") < 0)
						tempSampleCount -= 1;

					if (tempSampleCount < minSamples)
						tempSampleCount = minSamples;
					else if(tempSampleCount > maxSamples)
						tempSampleCount = maxSamples;

					audioDirector.samplesPerDecadeArray[currentIndex] = tempSampleCount;
				}
				else // start sample index stuff
				{
					int tempSampleStartIndex = audioDirector.sampleStartIndex;

					if( Input.GetAxis("Editor Vertical") > 0)
						tempSampleStartIndex += 1;
					else if( Input.GetAxis("Editor Vertical") < 0)
						tempSampleStartIndex -= 1;

					if (tempSampleStartIndex < minSampleStartIndex)
						tempSampleStartIndex = minSampleStartIndex;
					else if(tempSampleStartIndex > maxSampleStartIndex)
						tempSampleStartIndex = maxSampleStartIndex;

					audioDirector.sampleStartIndex = tempSampleStartIndex;
				}



				incrementCooldownCounter = 0;
			}
			else
				incrementCooldownCounter += Time.deltaTime;


			incrementHeldDownDurationCounter += Time.deltaTime;
			

		}
		else
		{
			incrementCooldownCounter += Time.deltaTime;
			incrementHeldDownDurationCounter = 0;
		}


	}

	void AdjustRangeMarkerScale()
	{
		if(currentIndex != -1)
		{
			float tempMax = 0;
			float sum = 0;
			foreach( float scale in audioDirector.samplesPerDecadeArray)
			{
				if(scale > tempMax)
					tempMax = scale;
				sum += scale;
			}
			float average = sum/10.0f;

			float maxValue = tempMax;
			float scaleRatio = 0.5f * Mathf.Sqrt( audioDirector.samplesPerDecadeArray[currentIndex]/average ); //average;//maxValue;

			Vector3 markerScale = rangeMarker.transform.localScale;
			markerScale.y = scaleRatio;
			rangeMarker.transform.localScale = markerScale;
		}
		else
		{
			Vector3 markerScale = rangeMarker.transform.localScale;
			markerScale.y = audioDirector.sampleStartIndex/40.0f;
			rangeMarker.transform.localScale = markerScale;
		}
	}

	// THANK YOU --> // http://answers.unity3d.com/questions/157940/getoutputdata-and-getspectrumdata-they-represent-t.html
	void CalculateDecadeFrequencyRange(int decadeIndex, out float minFrequency, out float maxFrequency)
	{
		int decadeMinBinIndex = 0;
		int decadeMaxBinIndex; // should be more than 0

		// should never enter the loop if decadeIndex == 0, should leave decadeMinBinIndex as 0
		for(int i = 0; i < decadeIndex; i++)
			decadeMinBinIndex += 10 * audioDirector.samplesPerDecadeArray[i] + audioDirector.sampleStartIndex; // multiplied by 10 because of the decade system

		// calculate max bin index 
		decadeMaxBinIndex = decadeMinBinIndex + 10 * audioDirector.samplesPerDecadeArray[decadeIndex];

		float sampleCount = (float)audioDirector.sampleArrayFreqBH.Length;
		float sampleRate = (float)audioDirector.currentSampleRate;

		minFrequency = ( (float)decadeMinBinIndex * sampleRate/2.0f )/sampleCount;
		maxFrequency = ( (float)decadeMaxBinIndex * sampleRate/2.0f )/sampleCount;
	}


}
