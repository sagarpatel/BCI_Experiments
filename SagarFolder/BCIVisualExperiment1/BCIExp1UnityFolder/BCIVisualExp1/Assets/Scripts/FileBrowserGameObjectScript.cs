using UnityEngine;
using System.Collections;
using System.IO;

public class FileBrowserGameObjectScript : MonoBehaviour 
{

    // From: http://wiki.unity3d.com/index.php?title=ImprovedFileBrowser

    protected FileBrowser m_fileBrowser;
    protected Texture2D m_directoryImage, m_fileImage;

	
	string filePathmp3;
	string filePathtxt;

	public bool isActive = true;

	string lastUsedDirectorymp3 = null;
	string lastUsedDirectorytxt = null;
	string mostRecentExtension = null;

	//MP3Import mp3Importer;
	AudioDirectorScript audioDirector;
	AudioListener audioListener;
	GeneralEditorScript genralEditorScript;

	// Use this for initialization
	void Start () 
	{
		//mp3Importer = (MP3Import)GetComponent("MP3Import");
		audioDirector = (AudioDirectorScript) GameObject.FindWithTag("AudioDirector").GetComponent("AudioDirectorScript");
		audioListener = (AudioListener) GameObject.FindWithTag("AudioDirector").GetComponent("AudioListener");
		genralEditorScript = (GeneralEditorScript)GetComponent("GeneralEditorScript");

		if(isActive)
		{
			Screen.showCursor = true;
			Screen.lockCursor = false;
		}
	}
	
	// Update is called once per frame
	void Update () 
	{

		// check input to activate button
		if(  Input.GetButtonDown("Display Music Browser Button") == true )
		{
			if(isActive)
			{
				isActive = false;
				Screen.showCursor = false;
				Screen.lockCursor = true;
			}
			else
			{
				isActive = true;
				genralEditorScript.isActive = false;
				Screen.showCursor = true;
				Screen.lockCursor = false;
			}
		}
	
	}


    protected void OnGUI () 
    {
    	if(isActive)
    	{
	        if (m_fileBrowser != null) 
	        {
	            m_fileBrowser.OnGUI();
	        } 
	        else 
	        {
	            OnGUIMain();
	        }
	    }
    }
 
    protected void OnGUIMain() 
    {
    	// mp3 file browsing
        GUILayout.BeginHorizontal();
        GUILayout.Label("Press button to select a mp3 file", GUILayout.Width(100));
        GUILayout.FlexibleSpace();
        GUILayout.Label(filePathmp3 ?? "none selected");
        if( GUILayout.Button("Browse mp3 file!", GUILayout.ExpandWidth(false)) ) 
        {
            m_fileBrowser = new FileBrowser( new Rect(100, 100, 600, 500), "Choose Music (mp3) File", FileSelectedCallback, lastUsedDirectorymp3);
            m_fileBrowser.SelectionPattern = "*.mp3";
            m_fileBrowser.DirectoryImage = m_directoryImage;
            m_fileBrowser.FileImage = m_fileImage;
            mostRecentExtension = "mp3";
        }
        GUILayout.EndHorizontal();

        // track parameters file broswing
        GUILayout.BeginHorizontal();
        GUILayout.Label("Press button to select a parameters file", GUILayout.Width(100));
        GUILayout.FlexibleSpace();
        GUILayout.Label(filePathtxt ?? "none selected");
        if( GUILayout.Button("Browse Parameters File!", GUILayout.ExpandWidth(false)) ) 
        {
            m_fileBrowser = new FileBrowser( new Rect(100, 100, 600, 500), "Choose Parameters File", FileSelectedCallback, lastUsedDirectorytxt);
            m_fileBrowser.SelectionPattern = "*.txt";
            m_fileBrowser.DirectoryImage = m_directoryImage;
            m_fileBrowser.FileImage = m_fileImage;
            mostRecentExtension = "txt";
        }
        GUILayout.EndHorizontal();

    }
 
    protected void FileSelectedCallback(string path, string directory) 
    {
        m_fileBrowser = null;
        
        
        // handle mp3 file
        if( mostRecentExtension == "mp3")
        {
        	lastUsedDirectorymp3 = directory;
        	filePathmp3 = path;

        	 if(filePathmp3 != null)
	        {
		        //mp3Importer = (MP3Import)GetComponent("MP3Import");
		        //mp3Importer.StartImport(filePathmp3);

		        audioDirector.audioSourceArray[0].Stop();
		        audioDirector.audioSourceArray[0] = null;

		        //audioDirector.audioSourceArray[0] = mp3Importer.audioSource;
		        audioDirector.audioSourceArray[0].Play();
		        audioDirector.currentlyPlayingFileName = Path.GetFileName(path);

		        // fix for low pass filter not working on loaded songs
		        audioListener.enabled = false;
		        audioListener.enabled = true;

		        // fixes memory leaked cause by unsed audio clips (occurs when loading new songs)
		        Resources.UnloadUnusedAssets();

		    }
        } // handle parameters file
        else if( mostRecentExtension == "txt")
        {
        	lastUsedDirectorytxt = directory;
        	filePathtxt = path;

        	// load parameters text file into string
        	string rawFileString = File.ReadAllText(filePathtxt);
        	string[] rawSplit = rawFileString.Split('|');
        	string amplitudeListString = rawSplit[1];
        	string frequencyStartString = rawSplit[2];
        	string frequencyListString = rawSplit[3];
        	string rgbColorScaleFactorsString = rawSplit[4];


        	// copy over amplitudes
        	for(int i = 0; i < audioDirector.scalingPerDecadeArray.Length; i++)
        		audioDirector.scalingPerDecadeArray[i] = float.Parse( amplitudeListString.Split(',')[i] );
     		
     		// copy over frequency start index
     		audioDirector.sampleStartIndex = int.Parse( frequencyStartString.Trim() );

     		// copy over frequency distribution
     		for(int i = 0; i < audioDirector.samplesPerDecadeArray.Length; i++)
     			audioDirector.samplesPerDecadeArray[i] = int.Parse( frequencyListString.Split(',')[i] );

     		// copy over rgb scale values
     		audioDirector.rScale = float.Parse( rgbColorScaleFactorsString.Split(',')[0] );
     		audioDirector.gScale = float.Parse( rgbColorScaleFactorsString.Split(',')[1] );
     		audioDirector.bScale = float.Parse( rgbColorScaleFactorsString.Split(',')[2] );

        }
 
        //isActive = false;       
    }


}
