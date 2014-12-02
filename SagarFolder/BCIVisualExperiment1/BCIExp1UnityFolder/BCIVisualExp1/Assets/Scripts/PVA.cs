﻿using UnityEngine;
using System.Collections;

public class PVA : MonoBehaviour 
{	
	public bool rotationPointsToCurrentVelocity = false;

	public Vector3 position;
	public Vector3 velocity;
	public Vector3 acceleration;
	public float velocityMagnitude;

	public Vector3 rotationalVelocity;
	public Vector3 rotationalAcceleration;

	public float zRotationVelocity;
	public float zRotationAcceleration;


	public bool isLinearDecay = false;
	public bool isAngularDecay = false;

	[Range(0.0f,1.0f)]
	public float linearVelocityDecay = 0;

	[Range(0.0f,1.0f)]
	public float angularVelocityDecay = 0;
	public Vector3 angularFrictionAxisToggle = Vector3.zero;

	public Space refrenceFrame = Space.World;
	Vector3 deltaPos;

	public float velocityKillThreashold = 0.0f;
	public float maxVelocityMagnitude = 300.0f;
	public float maxRotationalVelocityMagnitude = 400.0f;
	Vector3 deltaV;
	Vector3 previousV;

	public float timeStep = 1.0f/1000.0f;
	public float stepCounter = 0;
	public int loopCounter = 0;

	// Use this for initialization
	void Start () 
	{
		Init();
		//previousV = new Vector3(0, 0, 0);
	}

	void Init()
	{
		position = transform.position;
	}
	
	// Update is called once per frame
	void Update () 
	{

		float currentDT = Time.deltaTime;

		loopCounter = 0;
		while(stepCounter < currentDT)
		{
			CalculatePVA();
			stepCounter += timeStep;
			loopCounter ++;
		}
		stepCounter -= currentDT;
			
		transform.Translate(deltaPos, refrenceFrame);
		deltaPos = Vector3.zero;

		velocityMagnitude = velocity.magnitude;

	}

	private void CalculatePVA()
	{
		// do core PVA update

		velocity += acceleration * timeStep;
		velocity = Vector3.ClampMagnitude(velocity, maxVelocityMagnitude);
		deltaPos += (velocity + previousV) * 0.5f * timeStep;

		rotationalVelocity += rotationalAcceleration * timeStep;
		rotationalVelocity = Vector3.ClampMagnitude(rotationalVelocity, maxRotationalVelocityMagnitude);
		
		if(isLinearDecay)
		{
			// apply decay
			velocity -= linearVelocityDecay * velocity * timeStep;
		}
		

		// always do angular friction, just apply depending on toggle
		Vector3 amountToRemove =  new Vector3(angularFrictionAxisToggle.x * rotationalVelocity.x,
											 angularFrictionAxisToggle.y * rotationalVelocity.y,
											 angularFrictionAxisToggle.z * rotationalVelocity.z);
		amountToRemove *= angularVelocityDecay * timeStep;
		rotationalVelocity -= amountToRemove;			

		if( Mathf.Abs(velocity.x) <= velocityKillThreashold )
			velocity.x = 0;
		if( Mathf.Abs(velocity.y) <= velocityKillThreashold )
			velocity.y = 0;
		if( Mathf.Abs(velocity.z) <= velocityKillThreashold )
			velocity.z = 0;
		if( Mathf.Abs(zRotationVelocity) <= velocityKillThreashold )
			zRotationVelocity = 0;

		deltaV = velocity - previousV;
		previousV = velocity;


	}

	public void ResetPVA()
	{
		Init();
	}


}
