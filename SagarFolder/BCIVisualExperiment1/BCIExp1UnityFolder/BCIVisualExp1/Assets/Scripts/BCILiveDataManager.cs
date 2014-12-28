using UnityEngine;
using System.Collections;
using UnityOSC;
using System.Collections.Generic;

public class BCILiveDataManager : MonoBehaviour 
{
	void Start()
	{
		OSCHandler.Instance.Init();

	}

	void Update()
	{

		OSCHandler.Instance.UpdateLogs();
		Dictionary<string, ServerLog> serversDict = new Dictionary<string, ServerLog>();
		serversDict = OSCHandler.Instance.Servers;

		Debug.Log("Looping through servers");
		ServerLog tempServerLog = new ServerLog();
		foreach( string serverKey in serversDict.Keys)
		{
			serversDict.TryGetValue(serverKey, out tempServerLog);
			Debug.Log(tempServerLog.log[0]);
		}

	}

}
